import math
import gym 
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering


class MazeEnv(gym.Env):
    def __init__(self, task={}):
        super(MazeEnv, self).__init__()
        # 0-up 1-down 2-left 3-right
        self.action_space = [0, 1, 2, 3]
        self.action_dim = len(self.action_space)
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(4)
        self.all_pos = [[i, j] for i in range(8) for j in range(8)]
        self.all_possible_traps = [[i, j] for i in range(1, 7) for j in range(1,7)]
        trap_index = np.random.randint(0, 36, 2)
        traps = [self.all_possible_traps[trap_index[0]], self.all_possible_traps[trap_index[1]]]
        self.all_possible_goal = [x for x in self.all_pos if x not in traps]
        goal_index = np.random.randint(0, len(self.all_possible_traps))
        goal = self.all_possible_goal[goal_index]

        self.viewer = None

        # Set task and traps
        self._task = task
        self._trap = task.get('traps', traps)
        self._goal = task.get('goal', goal)
        self._state = np.zeros(2, dtype=np.int32)
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _out_of_maze(self, action):
        if action == 0: # up
            if self._state[1] + 1 > 7:
                return True
            else: 
                return False
        if action == 1: # Down
            if self._state[1] - 1 < 0:
                return True
            else: 
                return False
        if action == 2: # Left
            if self._state[0] - 1 < 0:
                return True
            else: 
                return False
        if action == 3: # Right
            if self._state[0] + 1 > 7:
                return True
            else: 
                return False
    
    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']
        self._trap = task['traps']
    
    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.int32)
        return self._state
    
    def sample_task(self, num_tasks):


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        if not self._out_of_maze(action):
            if action == 0: # Up
                next_state = [self._state[0], self._state[1]+1]
            elif action == 1: # Down
                next_state = [self._state[0], self._state[1]-1]
            elif action == 2: # Left
                next_state = [self._state[0]-1, self._state[1]]
            elif action == 3: # Right
                next_state = [self._state[0]+1, self._state[1]]

            print ('next_state: {}, goal: {}, traps: {}'.format(next_state, self._goal, self._trap))
            
            if next_state == self._goal:
                print ('Congratz!')
                reward = 0
                done = True
            elif next_state in self._trap:
                reward = -16 - (abs(next_state[0] - self._goal[0]) + abs(next_state[1] - self._goal[1])) 
                print ('ooooops! There is a trap here, reward: ', reward)
                done = False
            else:
                reward = -(abs(next_state[0] - self._goal[0]) + abs(next_state[1] - self._goal[1]))
                done = False
            
            self._state = next_state
            return np.array(self._state), reward, done, {}
        else:
            print("hold current position ({} {})".format(self._state[0], self._state[1]))
            self._state = self._state
            reward = -(abs(self._state[0] - self._goal[0]) + abs(self._state[1] - self._goal[1]))
            done = False
            return np.array(self._state), reward, done, {}
    
    def render(self, mode='rgb_array'):
        width = 880
        height = 880

        if self.viewer is None:
            # Initialize Viewer
            self.viewer = rendering.Viewer(width, height)
            line_1 = rendering.Line((110,0), (110,880))
            line_1.set_color(0,0,0)
            line_2 = rendering.Line((220,0), (220,880))
            line_2.set_color(0,0,0)
            line_3 = rendering.Line((330,0), (330,880))
            line_3.set_color(0,0,0)
            line_4 = rendering.Line((440,0), (440,880))
            line_4.set_color(0,0,0)
            line_5 = rendering.Line((550,0), (550,880))
            line_5.set_color(0,0,0)
            line_6 = rendering.Line((660,0), (660,880))
            line_6.set_color(0,0,0)
            line_7 = rendering.Line((770,0), (770,880))
            line_7.set_color(0,0,0)
            line_8 = rendering.Line((0,110), (880,110))
            line_8.set_color(0,0,0)
            line_9 = rendering.Line((0,220), (880,220))
            line_9.set_color(0,0,0)
            line_10 = rendering.Line((0,330), (880,330))
            line_10.set_color(0,0,0)
            line_11 = rendering.Line((0,440), (880,440))
            line_11.set_color(0,0,0)
            line_12 = rendering.Line((0,550), (880,550))
            line_12.set_color(0,0,0)
            line_13 = rendering.Line((0,660), (880,660))
            line_13.set_color(0,0,0)
            line_14 = rendering.Line((0,770), (880,770))
            line_14.set_color(0,0,0)

            self.viewer.add_geom(line_1)
            self.viewer.add_geom(line_2)
            self.viewer.add_geom(line_3)
            self.viewer.add_geom(line_4)
            self.viewer.add_geom(line_5)
            self.viewer.add_geom(line_6)
            self.viewer.add_geom(line_7)
            self.viewer.add_geom(line_8)
            self.viewer.add_geom(line_9)
            self.viewer.add_geom(line_10)
            self.viewer.add_geom(line_11)
            self.viewer.add_geom(line_12)
            self.viewer.add_geom(line_13)
            self.viewer.add_geom(line_14)

            # Create goal point
            goal_point = rendering.make_circle(20)
            goal_trans = rendering.Transform(translation=(self._goal[0]*110+55, self._goal[1]*110+55))
            goal_point.add_attr(goal_trans)
            goal_point.set_color(1, 0.84, 0)
            self.viewer.add_geom(goal_point)

            # Create trap points
            trap_point_1 = rendering.make_circle(20)
            trap_trans_1 = rendering.Transform(translation=(self._trap[0][0]*110+55, self._trap[0][1]*110+55))
            trap_point_1.add_attr(trap_trans_1)
            trap_point_1.set_color(0,0,0)
            self.viewer.add_geom(trap_point_1)

            trap_point_2 = rendering.make_circle(20)
            trap_trans_2 = rendering.Transform(translation=(self._trap[1][0]*110+55, self._trap[1][1]*110+55))
            trap_point_2.add_attr(trap_trans_2)
            trap_point_2.set_color(0,0,0)
            self.viewer.add_geom(trap_point_2)

            # Create current state
            self.state_point = rendering.make_circle(20)
            self.state_trans = rendering.Transform(translation=(self._state[0]*110+55, self._state[1]*110+55))
            self.state_point.add_attr(self.state_trans)
            self.state_point.set_color(0.25, 0.42, 0.88)
            self.viewer.add_geom(self.state_point)
        # Create current state
        self.state_trans.set_translation(self._state[0]*110+55, self._state[1]*110+55)
        self.state_point.set_color(0.25, 0.42, 0.88)
        self.viewer.add_geom(self.state_point)
        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

            
