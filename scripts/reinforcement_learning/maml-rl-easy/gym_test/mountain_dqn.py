import gym 
import random
import time
import numpy as np
import pickle
import tensorflow as tf
from collections import deque

'''
Replace Q-Table with deep neural network
'''

class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 200
        # Size of training set
        self.replay_size = 2000
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()
    
    def create_model(self):
        '''
        Create a neural network with hidden size 100
        '''
        STATE_DIM = 2
        ACTION_DIM = 3
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(100, input_shape=(STATE_DIM,), activation='relu'))
        model.add(tf.keras.layers.Dense(ACTION_DIM, activation='linear'))
        model.compile(loss='mse', optimizer='adam')

        return model
    
    def act(self, s, epsilon=0.1):
        '''
        Predict actions using neural network
        '''
        # Introduce randomness firstly
        if np.random.uniform() < epsilon - self.step*0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])
    
    def save_model(self, file_path='./model/MountainCar-v0-dqn.h5'):
        print ('Model saved')
        self.model.save(file_path)
    
    def remember(self, s, a, next_s, reward):
        '''
        For goal[0] = 0.5
        if next_s[0] > 0.4 give extra reward to boost the train process
        '''
        if next_s[0] > 0.4:
            reward += 1
        self.replay_queue.append((s, a, next_s, reward))
    
    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return 
        self.step += 1

        # Every update_freq, update the weights of self.model to self.target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # Update Q value in training set
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))
        
        # Input data to neural network
        self.model.fit(s_batch, Q, verbose=0)

    
env = gym.make('MountainCar-v0')
episodes = 1000
score_list = []
agent = DQN()

for i in range(episodes):
    s = env.reset()
    score = 0
    while True:
        a = agent.act(s)
        next_s, reward, done, _ = env.step(a)
        agent.remember(s, a, next_s, reward)
        agent.train()
        score += reward
        s = next_s

        if done:
            score_list.append(score)
            print ('Episode: {}, score: {}, currently max score: {}'.format(i, score, max(score_list)))
        
            break
    if np.mean(score_list[-10:]) > -180:
        agent.save_model()
        break
env.close()
    