import gym
import numpy as np
import argparse


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--th", type=float, default=1e-5)
parser.add_argument("--test_time", type=int, default=1000)

arg = parser.parse_args()
# env = gym.make('FrozenLake8x8-v0')
env = gym.make("Taxi-v3")
policy = np.random.uniform(0, 1.0, (env.nS, env.nA))
for _ in range(100):
    V = np.zeros(env.nS)
    for _ in range(100):
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + arg.gamma * V[next_state])
            V[s] = v

    for _ in range(100):
        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += reward + arg.gamma * prob * V[next_state]
            temp = np.zeros(env.nA)
            temp[np.where(action_values == np.max(action_values))] = 1.
            # temp =
            policy[s] = temp / temp.sum()

win_time = 0
for _ in range(arg.test_time):
    obs = env.reset()
    while True:
        action = np.argmax(policy[obs])
        obs, reward, done, info = env.step(action)
        if done:
            if reward == 20.:
                win_time += 1
            break
print(win_time / arg.test_time)
