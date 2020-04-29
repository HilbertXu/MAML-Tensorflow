# -*- coding: utf-8 -*-
import scipy.misc
import gym
'''
MountainCar-V0
========================================
State: Position, Speed
Action: 0-left 1-hold 2-right
Reward: -1 per round
Done: 200 round in total/Reach the peak
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

STATE_DIM = 2
ACTION_DIM = 3

# Set the precsion of keras otherwise the sum of probability given by softmax will not be 1
tf.keras.backend.set_floatx('float64')

class PGModel(tf.keras.models.Model):
    def __init__(self, input_dim, output_dim):
        super(PGModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize layers
        self.dense_1 = tf.keras.layers.Dense(128, input_shape=(None,self.input_dim), activation='relu')
        # tf.keras.layers.Dropout(0.1)
        self.all_act = tf.keras.layers.Dense(self.output_dim)
    
    def call(self, state):
        x = self.dense_1(state)
        x = self.all_act(x)
        self.logits = x
        output = tf.keras.activations.softmax(x)
        #output = tf.nn.softmax(x)
        return output, self.logits
    
class PolicyGradient(object):
    def __init__(
        self,
        lr = 0.001,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        reward_decay=0.95
    ):
        # Learning rate
        self.lr = lr
        # Dimension of state space
        self.state_dim = state_dim
        # Dimension of action space
        self.action_dim = action_dim
        # reward decay rate
        self.reward_decay = reward_decay
        # Observation, actions, reward of an episode
        self.ep_obs, self.ep_acts, self.ep_rs = [], [], []
        # Policy Net
        self.model = PGModel(STATE_DIM, ACTION_DIM)
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
    
    def loss_func(self, predict, actions, ep_rs_norm):
        actions = tf.one_hot(self.ep_acts, depth=self.action_dim)
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=actions)
        loss = tf.reduce_mean(neg_log_prob*ep_rs_norm)
        return loss
    
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_acts.append(a)
        self.ep_rs.append(r)

    def choose_action(self, state):
        prob_dist, _ = self.model(np.array([state]))
        action = np.random.choice(len(prob_dist[0]), p=prob_dist[0])
        return action
    
    def discount_and_norm_reward(self):
        out = np.zeros_like(self.ep_rs)
        dis_reward = 0 

        # Calculate reward with discount
        for i in reversed(range(len(self.ep_rs))):
            dis_reward = dis_reward +  self.reward_decay * self.ep_rs[i]
            out[i] = dis_reward
        # Normalization
        out -= np.mean(out)
        out /= np.std(out)
        return out
    
    def train_op(self):
        discounted_reward = self.discount_and_norm_reward()

        with tf.GradientTape() as tape:
            prob_dist, logits = self.model(np.vstack(self.ep_obs))
            loss = self.loss_func(logits, self.ep_acts, discounted_reward)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.ep_obs, self.ep_acts, self.ep_rs = [], [], []




# Use default parameters
agent_pg = PolicyGradient()
# Make gym environment
env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped
DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

'''
此处训练时遇到了Policy Gradient的一个主要问题
'''

for ep_idx in range(2000):
    observation = env.reset()
    while True:
        if RENDER:
            env.render()
        # Choose action with current policy
        action = agent_pg.choose_action(observation)
        # Execute the action
        _obs, reward, done, info = env.step(action)
        # Store ob, action, reward
        agent_pg.store_transition(_obs, action, reward)
        # Update observation
        observation = _obs

        if done:
            ep_rs_sum = sum(agent_pg.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            print ('Episode: {} Reward: {}'.format(ep_idx, int(running_reward)))

            # Update parameters of policy using the policy gradient
            agent_pg.train_op()

            break
        
            




        

    

        
        



