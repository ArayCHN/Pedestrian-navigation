import gym
from gym import error, spaces, utils
from gym.utils import seeding

class CrosswalkEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print("initialized environment!")
    pass
  def step(self, action):
    pass
  def reset(self):
    pass
  def render(self, mode='human'):
    pass
  def close(self):
    pass
