import gym 
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

'''
CartPole-V0
=========================================================
State: car speed, car position, bar speed, bar position
Action: 0-move left, 1-move right
Reward: 1 per step
Done: 200 round in total
'''
env = gym.make('CartPole-v0')

STATE_DIM = 4
ACTION_DIM = 2

model = tf.keras.Sequential()
model.add(Dense(64, input_shape=(STATE_DIM,), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(ACTION_DIM, activation='linear'))

model.summary()

def generate_episode_data():
    x, y, score = [], [], 0
    state = env.reset()

    while True:
        action = random.randrange(0, 2)
        x.append(state)
        y.append([1, 0] if action==0 else [0,1])
        state, reward, done, _ = env.step(action)
        score += reward

        if done:
            break
    
    return x, y, score


def generate_training_data(min_score=100):
    '''
    Generate N episodes, use episodes with score>100 as training data
    '''
    data_x, data_y, scores = [], [], []

    for i in range(10000):
        x, y, score = generate_episode_data()

        if score > min_score:
            data_x.extend(x)
            data_y.extend(y)
            scores.append(score)
    
    print ('dataset size: {}, max score: {}'.format(len(data_x), max(scores)))
    return np.array(data_x), np.array(data_y)


data_x, data_y = generate_training_data()
print(data_x.shape)
model.compile(loss='mse', optimizer='adam')
model.fit(data_x, data_y, epochs=5)
model.save('./model/CartPole-v0-nn.h5')

for i in range(5):
    state = env.reset()
    scores = 0
    while True:
        time.sleep(0.1)
        env.render()
        action = np.argmax(model.predict(np.array([state]))[0])
        state, reward, done, _ = env.step(action)
        scores += reward
        if done:
            print('CartPole Using NN, final score: {}'.format(scores))
            break

env.close()
