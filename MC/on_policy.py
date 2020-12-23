import gym
import numpy as np
import argparse
from tqdm import trange

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--th", type=float, default=1e-5)
parser.add_argument("--test_time", type=int, default=100)
parser.add_argument("--episode_num", type=int, default=2000)
parser.add_argument("--epsilon", type=float, default=0.3)

arg = parser.parse_args()

env = gym.make('FrozenLake8x8-v0')
policy = np.random.uniform(0, 1.0, (env.nS, env.nA))
action_values = np.random.uniform(0, 1.0, (env.nS, env.nA))


def sample():
    global env, policy
    episodes = list()  # arg.episode_num, 100, 3
    for e in range(arg.episode_num):
        state = env.reset()
        episode = list()
        for i in range(100):
            action_prob = policy[state]
            best_action = np.argmax(action_prob)
            action_prob = np.ones_like(action_prob) * arg.epsilon / env.nA
            action_prob[best_action] += 1.0 - arg.epsilon
            action = np.random.choice(np.arange(env.nA), p=action_prob)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        episodes.append(episode)
    return episodes


for _ in trange(10):
    episodes = sample()
    action_values_times = np.zeros_like(action_values)
    for episode in episodes:
        for i, (state, action, reward) in enumerate(episode):
            episode_array = np.asarray(episode)
            action_value = np.power(np.ones_like(episode_array[1:, 2])*arg.gamma, np.arange(len(episode_array[1:, 2])))
            action_value = action_value * episode_array[1:, 2]
            action_value = action_value.mean()
            n = action_values_times[state][action]
            action_values[state][action] = (action_values[state][action] * n + action_value) / (n+1)
            action_values_times[state][action] += 1.

    for s in range(env.nS):
        action_value = action_values[s]
        policy[s] = np.zeros(env.nA)
        policy[s][np.argmax(action_value)] = 1.

win_time = 0
for _ in trange(arg.test_time):
    obs = env.reset()
    while True:
        action = np.argmax(policy[obs])
        obs, reward, done, info = env.step(action)
        if done:
            if reward == 1:
                win_time += 1
            break
print(win_time / arg.test_time)