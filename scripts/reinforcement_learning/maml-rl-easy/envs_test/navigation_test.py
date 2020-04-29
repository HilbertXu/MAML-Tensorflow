import gym
import time
import random
import numpy as np

env = gym.make('Navigation2D-v0')

state = env.reset()

goals = np.random.uniform(-0.5, 0.5, size=(2,))
task = {'goal': goals}

env.reset_task(task)

score = 0

# Without any policy
while True:
    time.sleep(1)
    env.render()
    action = np.random.uniform(-0.1, 0.1, size=(2,))
    state, reward, done, _ = env.step(action)
    score += reward
    if done:       # 游戏结束
        print('score: ', score)  # 打印分数
        break
env.close()
