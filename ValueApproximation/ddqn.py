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
parser.add_argument("--init_epsilon", type=float, default=0.1)
parser.add_argument("--final_epsilon", type=float, default=1e-3)
parser.add_argument("--explore_num", type=int, default=20000)
parser.add_argument("--test_time", type=int, default=100)
parser.add_argument("--episode_num", type=int, default=40000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lambda_", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--replay_buffer", type=int, default=4000)
parser.add_argument("--update_time", type=int, default=100)

arg = parser.parse_args()

env = gym.make('CartPole-v0')

q_value_network = keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(env.action_space.n),
])
q_target_network = keras.models.clone_model(q_value_network)
opt = keras.optimizers.Adam(learning_rate=arg.lr)
replay_memory = deque()
epsilon = arg.init_epsilon
time_step = 0


def copy():
    global q_value_network, q_target_network
    for wt, w in zip(q_target_network.weights, q_value_network.weights):
        wt.assign(w)


@tf.function
def train_step(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
    q_target_batch = q_target_network(next_state_batch)
    q_value_batch = q_value_network(next_state_batch)
    max_action_next = tf.math.argmax(q_value_batch, axis=-1)
    # TODO
    max_action_next = tf.one_hot(max_action_next, env.action_space.n)
    target_value_batch = tf.einsum("bi,bi->b", q_target_batch, max_action_next)
    y_batch = reward_batch + (1 - terminal_batch) * arg.gamma * target_value_batch
    with tf.GradientTape() as tape:
        q_value_batch = q_value_network(state_batch)
        action_batch = tf.one_hot(action_batch, env.action_space.n)
        q_action = tf.reduce_sum(q_value_batch * action_batch, axis=1)
        loss = keras.losses.MSE(y_batch, q_action)
    gradients = tape.gradient(loss, q_value_network.trainable_variables)
    opt.apply_gradients(zip(gradients, q_value_network.trainable_variables))
    return loss


for episode in range(arg.episode_num):
    state = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    while True:
        q_value = q_value_network(state.reshape(1, -1)).numpy().squeeze()
        best_action = np.argmax(q_value)
        action_prob = np.ones_like(q_value) * epsilon / env.action_space.n
        action_prob[best_action] += 1.0 - epsilon
        action = np.random.choice(np.arange(env.action_space.n), p=action_prob)

        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        replay_memory.append((state, action, reward, next_state, int(done)))
        if len(replay_memory) >= arg.replay_buffer:
            replay_memory.popleft()
        if len(replay_memory) >= arg.batch_size:
            batch = random.sample(replay_memory, arg.batch_size)
            state_batch = np.ndarray(shape=(arg.batch_size, *state.shape), dtype=np.float32)
            action_batch = np.ndarray(shape=(arg.batch_size,), dtype=np.int32)
            reward_batch = np.ndarray(shape=(arg.batch_size,), dtype=np.float32)
            next_state_batch = np.ndarray(shape=(arg.batch_size, *state.shape), dtype=np.float32)
            terminal_batch = np.ndarray(shape=(arg.batch_size,), dtype=np.float32)
            for i, (s, a, r, ns, d) in enumerate(batch):
                state_batch[i] = s
                action_batch[i] = a
                reward_batch[i] = r
                next_state_batch[i] = ns
                terminal_batch[i] = d

            train_step(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

        state = next_state
        if epsilon <= arg.final_epsilon:
            epsilon -= (arg.init_epsilon - arg.final_epsilon) / arg.explore_num
        if time_step % arg.update_time == 0:
            if episode == 0:
                q_target_network.build(state.reshape(1, -1).shape)
            copy()
        time_step += 1
        if done:
            break
    if episode % 100 == 0:
        print(episode_reward)
        episode_rewards.append(episode_reward)
