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

network = keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(env.action_space.n, activation="softmax"),
])
opt = keras.optimizers.Adam(learning_rate=arg.lr)


@tf.function
def train_step(state_batch, action_batch, reward_batch):
    with tf.GradientTape() as tape:
        action_prob = network(state_batch)
        action_batch = tf.one_hot(action_batch, env.action_space.n)
        neg_log_prob = tf.reduce_sum(-tf.math.log(action_prob)*action_batch, axis=-1)
        loss = tf.reduce_mean(neg_log_prob * reward_batch)
    gradients = tape.gradient(loss, network.trainable_variables)
    opt.apply_gradients(zip(gradients, network.trainable_variables))
    # return loss


for episode in range(arg.episode_num):
    state = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    states = list()
    rewards = list()
    actions = list()
    while True:
        action_prob = network(state.reshape(1, -1)).numpy().squeeze()
        action = np.random.choice(range(env.action_space.n), p=action_prob)
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        state = next_state
        if done:
            states_array = np.asarray(states).astype(np.float32)
            actions_array = np.asarray(actions).astype(np.int32)
            rewards_array = np.asarray(rewards).astype(np.float32)

            discounted_array = np.zeros_like(rewards_array)
            running_add = 0
            for t in reversed(range(0, len(rewards_array))):
                running_add = running_add * arg.gamma + rewards_array[t]
                discounted_array[t] = running_add
            discounted_array -= np.mean(discounted_array)
            discounted_array /= np.std(discounted_array)
            
            train_step(states_array, actions_array, discounted_array)
            break
    if episode % 100 == 0:
        print(episode_reward)
        episode_rewards.append(episode_reward)
