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

arg = parser.parse_args()

env = gym.make('CartPole-v0')

actor_network = keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(env.action_space.n, activation="softmax"),
])
actor_opt = keras.optimizers.Adam(learning_rate=arg.lr)

critic_network = keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(1),
])
critic_opt = keras.optimizers.Adam(learning_rate=arg.lr)


@tf.function
def critic_train_step(state_batch, reward_batch, next_state_batch):
    v_ = critic_network(next_state_batch)
    with tf.GradientTape() as tape:
        td_error = reward_batch + arg.gamma * v_ - critic_network(state_batch)
        loss = tf.math.square(td_error)
    gradients = tape.gradient(loss, critic_network.trainable_variables)
    critic_opt.apply_gradients(zip(gradients, critic_network.trainable_variables))
    return td_error


@tf.function
def actor_train_step(state_batch, action_batch, td_error_batch):
    with tf.GradientTape() as tape:
        action_prob = actor_network(state_batch)
        log_prob = tf.math.log(tf.reduce_sum(action_prob * tf.one_hot(action_batch, env.action_space.n)))
        loss = - log_prob * td_error_batch
    gradients = tape.gradient(loss, actor_network.trainable_variables)
    actor_opt.apply_gradients(zip(gradients, actor_network.trainable_variables))


for episode in range(arg.episode_num):
    state = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    while True:
        state = state.reshape(1, -1)
        action_prob = actor_network(state).numpy().squeeze()
        action = np.random.choice(range(env.action_space.n), p=action_prob)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape(1, -1)

        td_error = critic_train_step(state, reward, next_state)
        actor_train_step(state, action, td_error)
        episode_reward += reward
        state = next_state
        if done:
            break
    if episode % 100 == 0:
        print(episode_reward)
        episode_rewards.append(episode_reward)
