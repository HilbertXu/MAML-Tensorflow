import gym 
import random
import time
import numpy as np
import pickle
from collections import defaultdict

'''
MountainCar-V0
========================================
State: Position, Speed
Action: 0-left 1-hold 2-right
Reward: -1 per round
Done: 200 round in total/Reach the peak
'''
env = gym.make('MountainCar-v0')

'''
Q-learning update rule
=======================================================================
Q[s][a] = (1 - lr) * Q[s][a] + lr * (reward + factor * max(Q[next_s]))
=======================================================================
s, a, next_s: current state, action, next state
reward: reward for acting actions
Q[s][a]: the quality of action a in state s
max(Q[next_s]): the maximum quality of all actions in next state next_s
lr: learning rate, bigger lr for less history experience
factor: discount factor, bigger factor for more history experience
'''

# Initialize Q-Table
Q = defaultdict(lambda: [0, 0, 0])


def transform_state(state):
    '''
    transform continous State(position, speed) to discrete state(40x40)
    '''
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    pos_int = 40 * (pos - pos_low) / (pos_high - pos_low)
    v_int = 40 * (v - v_low) / (v_high - v_low)

    return int(pos_int), int(v_int)


lr, factor = 0.7, 0.55
episodes = 10000
score_list = []

for i in range(episodes):
    s = transform_state(env.reset())
    score = 0
    while True:
        a = np.argmax(Q[s])
        # Introduce more randomness
        if np.random.random() > i*3 / episodes:
            a = np.random.choice([0, 1, 2])
        # Apply actions
        next_s, reward, done, _ = env.step(a)
        next_s = transform_state(next_s)
        # Update Q-Table
        Q[s][a] = (1-lr)*Q[s][a] + lr*(reward + factor*max(Q[next_s]))
        score += reward

        s = next_s

        if done:
            score_list.append(score)
            print ('Episode: {}, score: {}, currently max score: {}'.format(i, score, max(score_list)))
            break

env.close()

with open('./model/MountainCar-v0-q-learning.pickle', 'wb') as f:
    pickle.dump(dict(Q), f)
    print('model saved')

s = env.reset()
score = 0
while True:
    env.render()
    time.sleep(0.01)
    # transform_state函数 与 训练时的一致
    s = transform_state(s)
    a = np.argmax(Q[s]) if s in Q else 0
    s, reward, done, _ = env.step(a)
    score += reward
    if done:
        print('score:', score)
        break
env.close()




