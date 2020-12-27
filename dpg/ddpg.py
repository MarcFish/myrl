import gym
import numpy as np
import argparse
from tqdm import trange, tqdm
import random
import tensorflow as tf
import tensorflow.keras as keras
from collections import deque


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--test_time", type=int, default=100)
parser.add_argument("--episode_num", type=int, default=40000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--init_var", type=float, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--replay_buffer", type=int, default=4000)
parser.add_argument("--update_time", type=int, default=100)

arg = parser.parse_args()

env = gym.make('Pendulum-v0')

actor_opt = keras.optimizers.Adam(learning_rate=arg.lr)
critic_opt = keras.optimizers.Adam(learning_rate=arg.lr)
replay_memory = deque()
time_step = 0
var = arg.init_var


def create_actor_network():
    state_input = keras.layers.Input(env.observation_space.shape)
    network = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(env.action_space.shape[0], activation="tanh"),
    ])
    o = network(state_input)
    o = tf.math.multiply(o, env.action_space.high)
    return keras.Model(inputs=state_input, outputs=o)


def create_critic_network():
    state_input = keras.layers.Input(env.observation_space.shape)
    action_input = keras.layers.Input(env.action_space.shape)
    state_network = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
    ])
    action_network = keras.Sequential([
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
    ])
    state_output = state_network(state_input)
    action_output = action_network(action_input)
    q = keras.layers.Dense(1, activation="relu")(keras.layers.Add()([state_output, action_output]))
    return keras.Model(inputs=[state_input, action_input], outputs=q)


actor_eval_network = create_actor_network()
critic_eval_network = create_critic_network()
actor_target_network = keras.models.clone_model(actor_eval_network)
critic_target_network = keras.models.clone_model(critic_eval_network)


def copy():
    global actor_eval_network, critic_eval_network, actor_target_network, critic_target_network
    for wt, w in zip(actor_target_network.weights, actor_eval_network.weights):
        wt.assign(wt*(1-arg.tau)+w*arg.tau)
    for wt, w in zip(critic_target_network.weights, critic_eval_network.weights):
        wt.assign(wt*(1-arg.tau)+w*arg.tau)


@tf.function
def train_step(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
    action_ = actor_target_network(next_state_batch)
    q_target = reward_batch + arg.gamma * (critic_target_network([next_state_batch, action_])) * (1 - terminal_batch)
    with tf.GradientTape() as tape:
        td_error = keras.losses.MSE(y_true=q_target, y_pred=critic_eval_network([state_batch, action_batch]))
    critic_gradients = tape.gradient(td_error, critic_eval_network.trainable_variables)
    critic_opt.apply_gradients(zip(critic_gradients, critic_eval_network.trainable_variables))
    with tf.GradientTape(persistent=True) as tape:
        action = actor_eval_network(state_batch)
        q = critic_eval_network([state_batch, action])
    gradients = tape.gradient(q, action)
    actor_gradients = tape.gradient(action, actor_eval_network.trainable_variables, -gradients)
    actor_opt.apply_gradients(zip(actor_gradients, actor_eval_network.trainable_variables))


for episode in range(arg.episode_num):
    state = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    while True:
        action = actor_eval_network(state.reshape(1, -1))
        action = np.clip(np.random.normal(action, var), env.action_space.low, env.action_space.high)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.squeeze()
        episode_reward += reward
        replay_memory.append((state, action, reward, next_state, int(done)))
        if len(replay_memory) >= arg.replay_buffer:
            replay_memory.popleft()
            var *= 0.995
        if len(replay_memory) >= arg.batch_size:
            batch = random.sample(replay_memory, arg.batch_size)
            state_batch = np.ndarray(shape=(arg.batch_size, *env.observation_space.shape), dtype=np.float32)
            action_batch = np.ndarray(shape=(arg.batch_size,), dtype=np.float32)
            reward_batch = np.ndarray(shape=(arg.batch_size,), dtype=np.float32)
            next_state_batch = np.ndarray(shape=(arg.batch_size, *env.observation_space.shape), dtype=np.float32)
            terminal_batch = np.ndarray(shape=(arg.batch_size,), dtype=np.float32)
            for i, (s, a, r, ns, d) in enumerate(batch):
                state_batch[i] = s
                action_batch[i] = a
                reward_batch[i] = r
                next_state_batch[i] = ns
                terminal_batch[i] = d

            train_step(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
        state = next_state
        if time_step % arg.update_time == 0:
            copy()
        time_step += 1
        if done:
            break
    if episode % 100 == 0:
        print(episode_reward)
        episode_rewards.append(episode_reward)
