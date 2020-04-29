import gym
import time
import random
import numpy as np

env = gym.make('Maze-v0')

state = env.reset()

action_list = ['up', 'down', 'left', 'right']

all_pos = [[i, j] for i in range(8) for j in range(8)]
all_possible_traps = [[i, j] for i in range(1, 7) for j in range(1,7)]
goal_index = np.random.randint(0, 64)
goals = all_pos[goal_index]
trap_index = np.random.randint(0, 36, 2)
traps = [all_possible_traps[trap_index[0]], all_possible_traps[trap_index[1]]]

task = {'goal': goals, 'traps':traps}

env.reset_task(task)

score = 0

# Without any policy
while True:
    time.sleep(1)
    env.render()
    action = np.random.randint(0, 4, 1)[0]
    print (action_list[action])
    state, reward, done, _ = env.step(action)
    score += reward
    print ('reward: ', reward, 'done: ', done)
    print ('=======================')
    if done:       # 游戏结束
        print('score: ', score)  # 打印分数
        break
env.close()
