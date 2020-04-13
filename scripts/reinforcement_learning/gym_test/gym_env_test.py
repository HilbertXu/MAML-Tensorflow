import gym
import time
import random

env = gym.make('MountainCar-v0')

state = env.reset()

score = 0

while True:
    time.sleep(0.1)
    env.render()
    action = random.randint(0, 2)
    state, reward, done, _ = env.step(action)
    score += reward
    if done:       # 游戏结束
        print('score: ', score)  # 打印分数
        break
env.close()