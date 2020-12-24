import gym
import numpy as np
import argparse
from tqdm import trange, tqdm


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--test_time", type=int, default=100)
parser.add_argument("--episode_num", type=int, default=40000)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lambda_", type=float, default=0.9)


arg = parser.parse_args()

# env = gym.make('FrozenLake8x8-v0')
env = gym.make("Taxi-v3")
policy = np.zeros((env.nS, env.nA))
action_values = np.zeros((env.nS, env.nA))
eligibility_trace = action_values.copy()


def choose_action(action_prob):
    best_action = np.argmax(action_prob)
    action_prob = np.ones_like(action_prob) * arg.epsilon / env.nA
    action_prob[best_action] += 1.0 - arg.epsilon
    action = np.random.choice(np.arange(env.nA), p=action_prob)
    return action, action == best_action


for _ in trange(arg.episode_num):
    state = env.reset()
    while True:
        action, _ = choose_action(action_values[state])
        next_state, reward, done, info = env.step(action)
        next_action, flag = choose_action(action_values[next_state])
        q_predict = action_values[state, action]
        if done:
            q_target = reward
        else:
            q_target = reward + arg.gamma * action_values[next_state, next_action]
        error = q_target - q_predict
        eligibility_trace[state, action] += 1.
        action_values += arg.lr * error * eligibility_trace
        if flag:
            eligibility_trace *= arg.gamma * arg.lambda_
        else:
            eligibility_trace *= 0
        if done:
            break
        state = next_state

for s in range(env.nS):
    action_prob = action_values[s]
    best_action = np.argmax(action_prob)
    policy[s] = np.zeros(env.nA)
    policy[s][best_action] = 1.

win_time = 0
for _ in trange(arg.test_time):
    obs = env.reset()
    while True:
        action = np.argmax(policy[obs])
        obs, reward, done, info = env.step(action)
        if done:
            if reward == 20.:
                win_time += 1
            break
print(win_time / arg.test_time)
