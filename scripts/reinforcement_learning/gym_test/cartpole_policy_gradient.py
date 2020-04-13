import gym
'''
CartPole-V0
=========================================================
State: car speed, car position, bar speed, bar position
Action: 0-move left, 1-move right
Reward: 1 per step
Done: 200 round in total
'''
env = gym.make('CartPole-v0')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

STATE_DIM = 4
ACTION_DIM = 2

model = tf.keras.Sequential()
model.add(Dense(100, input_shape=(STATE_DIM,), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(ACTION_DIM, activation='softmax'))

model.summary()
model.compile(loss='mse', optimizer='adam')

def choose_action(s):
    '''
    Choose action according to its probability
    '''
    prob = model.predict(np.array([s]))[0]
    return np.random.choice(len(prob), p=prob)

def reward_discount(rewards, gamma=0.95):
    '''
        calculate the rewards with discount ratio
        last step：1
        last-1 step：1 + 0.95 * 1 = 1.95
        last-2 step：1 + 0.95 * 1.95 = 2.8525
        last-3 step：1 + 0.95 * 2.8525 = 3.709875
        ...
    '''
    prior = 0
    # Out has the same shape og input rewards
    out = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        # Reverse all rewards
        prior = prior * gamma + rewards[i]
        out[i] = prior
    return out/np.std(out - np.mean(out))

def train(records):
    s_batch = np.array([record[0] for record in records])
    # action 独热编码处理，方便求动作概率，即 prob_batch
    a_batch = np.array([[1 if record[1] == i else 0 for i in range(ACTION_DIM)]
                        for record in records])
    # 假设predict的概率是 [0.3, 0.7]，选择的动作是 [0, 1]
    # 则动作[0, 1]的概率等于 [0, 0.7] = [0.3, 0.7] * [0, 1]
    prob_batch = model.predict(s_batch) * a_batch
    r_batch = discount_rewards([record[2] for record in records])

    model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)

episodes = 2000  # 至多2000次
score_list = []  # 记录所有分数
for i in range(episodes):
    s = env.reset()
    score = 0
    replay_records = []
    while True:
        a = choose_action(s)
        next_s, r, done, _ = env.step(a)
        replay_records.append((s, a, r))

        score += r
        s = next_s
        if done:
            train(replay_records)
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    # 最后10次的平均分大于 195 时，停止并保存模型
    if np.mean(score_list[-10:]) > 195:
        model.save('CartPole-v0-pg.h5')
        break
env.close()

plt.plot(score_list)
x = np.array(range(len(score_list)))
smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
plt.show()

