import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np
import os
import pickle
import pandas as pd

class CrosswalkEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CrosswalkEnv, self).__init__()
    # observation: the closest 3 people, each 1.0 / distance_x, 1.0 / distance_y, (intensity), velocity (vx, vy)
    # initialize some parameters
    self.MAX_TIME_STEP = 1500 # the max num of steps. if exceeds this value and agent hasn't reached end, force stop the episode.
    self.GOAL_THRESHOLD = 20.0 # goal region, how many pixels? This paramter is deprecated. useless.
    self.COLLISION_REWARD = -500.0
    self.TIME_REWARD = -1.0
    self.GOAL_REWARD = 500.0
    self.MAX_DISTANCE_INVERT = 1.0 / 7.0 # if distance < 7 * 1.414 = 10 pixels, we determine this as a collision
    self.MAX_VELOCITY = 3.0 # max velocity
    # action space has to be symmtric, add offset to enforce positive velocity later
    self.action_space = spaces.Box(low=-self.MAX_VELOCITY / 2.0, high=self.MAX_VELOCITY / 2.0, shape=(1,), dtype=np.float32) # must be symmetric for ddpg
    self.observation_space = spaces.Box(
      low=np.array([-self.MAX_DISTANCE_INVERT, -self.MAX_DISTANCE_INVERT, -self.MAX_VELOCITY, -self.MAX_VELOCITY] * 3), 
      high=np.array([self.MAX_DISTANCE_INVERT, self.MAX_DISTANCE_INVERT, self.MAX_VELOCITY, self.MAX_VELOCITY] * 3),
      dtype=np.float32)
    # preload data names
    # TODO load pickle files here!
    # suppose we have self.paths and self.videos
    # paths[i] --> path, path[i] --> (track_id, history), history[i] --> [frame_id, x, y, vx, vy]
    # videos[i] --> video, video[i] --> frames, frames[i] = frame, frame[i] = [track_id, x, y, vx, vy]
    with open('/home/cs238/baselines/crosswalk/crosswalk/envs/dataset_process/pickle_frames_mix.pickle', 'rb') as file:
      self.videos = pickle.load(file)
    with open('/home/cs238/baselines/crosswalk/crosswalk/envs/dataset_process/path_mix.pickle', 'rb') as file:
      self.paths = pickle.load(file)
    self.num_videos = len(self.videos)
    print("initialized environment!")
  
  def close_to_goal(self):
    return ((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2) ** 0.5 < self.GOAL_THRESHOLD

  def step(self, action):
    # if there is no collision, give a time penalty
    rew = self.TIME_REWARD
    info = {"x": self.x, "y":self.y, "goal":(self.goal_x, self.goal_y)} # dict, debug info, empty for now
    info["index"] = self.index # record the index of video to see the success_rate of each video
    info["success"] = 0.0
    # print(info)
    done = False
    self.time_step += 1
    info["time"] = self.time_step
    self.current_frame_id += 1
    v = action + self.MAX_VELOCITY / 2.0 # velocity norm, should be greater than 0.0
    v_preserve = v
    # v is the distance the agent travels in one time step

    # if time step exceeds limit, force terminate
    if self.time_step > self.MAX_TIME_STEP:
      done = True
      new_obs = np.zeros((12,))
      return new_obs, rew, done, info
    
    # update self.x, self.y, and determine if we have reached goal
    # keep updating v to be the distance left to travel
    x0, y0 = self.x, self.y
    x_before, y_before = x0, y0
    while self.ego_traj_index < self.ego_traj_length - 1:
      x1, y1 = self.ego_trajectory[self.ego_traj_index + 1][1:3]
      distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
      if distance < v:
        v -= distance
        x0, y0 = x1, y1
        self.ego_traj_index += 1
      elif distance == v:
        self.x, self.y = x1, y1
        self.ego_traj_index += 1
        break
      else: # > v, overshoots
        portion = (v + 0.0) / distance
        self.x, self.y = x0 + portion * (x1 - x0), y0 + portion * (y1 - y0)
        break
    if self.ego_traj_index >= self.ego_traj_length - 1: # has reached goal
      self.x, self.y = x1, y1
      done = True
      rew = self.GOAL_REWARD
      new_obs = np.zeros((12,))
      info["success"] = 1.0 # reached goal
      # info["time"] = self.time_step # total time consumed to reach the goal
      # print(rew)
      return new_obs, rew, done, info

    # TODO unsure if this is the correct way!
    if self.current_frame_id >= len(self.frames):
      # has reached the end of all frames, but hasn't reached goal yet. The the episode is not done,
      # just always assume the people around agent are from the last frame
      self.current_frame_id = len(self.frames) - 1

    vx = self.x - x_before
    vy = self.y - y_before
    new_obs = self.observe(self.frames[self.current_frame_id], (self.x, self.y, vx, vy))
    # determine if there is a collision
    if (abs(new_obs[0]) >= self.MAX_DISTANCE_INVERT and abs(new_obs[1]) >= self.MAX_DISTANCE_INVERT or
        abs(new_obs[4]) >= self.MAX_DISTANCE_INVERT and abs(new_obs[5]) >= self.MAX_DISTANCE_INVERT or
        abs(new_obs[8]) >= self.MAX_DISTANCE_INVERT and abs(new_obs[9]) >= self.MAX_DISTANCE_INVERT):
        # there is a collision
        rew = self.COLLISION_REWARD
        done = True
    # print(rew)
    return new_obs, rew, done, info

  def observe(self, frame, position):
    # TODO vectorize to save time!
    x0, y0, vx0, vy0 = position
    if vx0 == 0:
      vx0 = 0.01
    if vy0 == 0:
      vy0 = 0.01
    ans = []
    sign = lambda x: 1 if x >= 0 else -1
    inv = lambda x: 1.0 / x if abs(x) > 1.0 / self.MAX_DISTANCE_INVERT else sign(x) * self.MAX_DISTANCE_INVERT
    x_comp = vx0 / ((vx0 ** 2 + vy0 ** 2)**0.5)
    y_comp = vy0 / ((vx0 ** 2 + vy0 ** 2)**0.5) # normalized
    # new_ax_x = (x_comp, y_comp)
    # new_ax_y = (y_comp, -x_comp)
    for id, x, y, vx, vy in frame:
      if id != self.ego_id:
        dx = x - x0
        dy = y - y0
        dvx = vx - vx0
        dvy = vy - vy0
        dx_new = dx * x_comp + dy * y_comp # dx in ego frame
        dy_new = dx * y_comp - dy * x_comp
        dvx_new = dvx * x_comp + dvy * y_comp
        dvy_new = dvx * y_comp - dvy * x_comp
        ans.append([inv(dx_new), inv(dy_new), dvx_new, dvy_new]) # represented in ego frame
    ans.sort(key=lambda x: 1.0/x[0]**2 + 1.0/x[1]**2)
    while len(ans) < 3:
      ans.append([0.0, 0.0, 0.0, 0.0])
    ans = ans[:3]
    obs = np.array(ans).reshape((12,))
    return obs

  def reset(self):
    # TODO retrieve an experience from dataset and initialize the location of ego vehicle
    # randomly select an agent
    # obs = np.array([0.0] * 12) # wrong
    # randomly select a video and a start time
    index = np.random.choice(self.num_videos) # choose from 0 to num_videos - 1
    self.index = index # use info to debug
    path = self.paths[index]
    num_ids = len(path)
    track_id = np.random.choice(num_ids)
    # print(("reset!!!", track_id))
    self.ego_id = path[track_id][0][0] # ego id
    self.ego_trajectory = path[track_id][:, 1:] # history, history[i] = [frame_id, x, y, vx, vy]
    self.current_frame_id = int(self.ego_trajectory[0][0])
    self.frames = self.videos[index]
    self.time_step = 0
    self.x, self.y = self.ego_trajectory[self.time_step][1], self.ego_trajectory[self.time_step][2]
    self.goal_x, self.goal_y = self.ego_trajectory[-1][1], self.ego_trajectory[-1][2]
    self.ego_traj_length = len(self.ego_trajectory)
    self.ego_traj_index = 0 # starting from #0 waypoint in ego_traj
    obs = self.observe(self.frames[self.current_frame_id], self.ego_trajectory[self.time_step][1:])
    return obs # initial observation


  def render(self, mode='human'):
    # do not render anything
    pass

  def close(self):
    # TODO what is this function?
    pass
