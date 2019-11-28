import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np

class CrosswalkEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CrosswalkEnv, self).__init__()
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=1, dtype=np.float32) # must be symmetric for ddpg
    # observation: the closest 3 people, each 1.0 / distance_x, distance_y, (intensity), velocity (vx, vy)
    # TODO specify observation
    self.observation_space = spaces.Box(low=np.array([-10.0, -10.0, -1.0, -1.0, -10.0, -10.0, -1.0, -1.0, -10.0, -10.0, -1.0, -1.0]), 
                                       high=np.array([10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0]), dtype=np.float32)
    print("initialized environment!")

  def step(self, action):
    # TODO use actual steps
    info = {} # dict, debug info
    new_obs = self.observation_space.sample()
    rew = 0.0
    done = True
    return new_obs, rew, done, info

  def reset(self):
    # TODO retrieve an experience from dataset and initialize the location of ego vehicle
    # obs = np.array([0.0] * 12) # wrong
    obs = self.observation_space.sample()
    return obs # initial observation

  def render(self, mode='human'):
    # do not render anything
    pass

  def close(self):
    # TODO what is this function?
    pass
