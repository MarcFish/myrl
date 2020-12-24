import gym
import numpy as np
import argparse
from tqdm import trange, tqdm
import tensorflow as tf
import tensorflow.keras as keras
from collections import deque


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--test_time", type=int, default=100)
parser.add_argument("--episode_num", type=int, default=40000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lambda_", type=float, default=0.9)
parser.add_argument("--action_num", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--replay_buffer", type=int, default=4000)

arg = parser.parse_args()

# env = gym.make('FrozenLake8x8-v0')
env = gym.make('CartPole-v0')

q_value_network = keras.Sequential([
    keras.layers.Dense(32),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(32),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(32),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(arg.action_num),
])
q_target_network = keras.models.clone_model(q_value_network)
opt = keras.optimizers.Adam(learning_rate=arg.lr)

replay_memory = deque()


def copy():
    for wt, w in zip(q_target_network.weights, q_value_network.weights):
        wt.assign(w)


@tf.function
def train_step(state_batch, action_batch, reward_batch, terminal_batch):
    q_value_batch = q_target_network(state_batch)
    y_batch = reward_batch + terminal_batch * arg.gamma * tf.math.reduce_max(q_value_batch, axis=-1)
    with tf.GradientTape() as tape:
        q_value_batch = q_value_network(state_batch)
        q_action = tf.reduce_sum(q_value_batch * action_batch, axis=1)
        loss = keras.losses.MSE(y_batch, q_action)
    gradients = tape.gradient(loss, q_value_network.trainable_variables)
    opt.apply_gradients(zip(gradients, q_value_network.trainable_variables))
    return loss
