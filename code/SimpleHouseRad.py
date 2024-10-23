import energym
import gymnasium as gym
from gymnasium import spaces
from utils import normalize, unnormalize
import numpy as np
import time
import torch

import energym
import gymnasium as gym
from gymnasium import spaces
from utils import normalize, unnormalize
import numpy as np

class SimpleHouseRad(gym.Env):    
    def __init__(self, action_type="continuous", weather="CH_BS_Basel", start_day=1, start_month=1, year=2019, eval_mode=False, online=False):
        self.env = energym.make('SimpleHouseRad-v0', weather=weather, start_day=start_day, start_month=start_month, year=year, simulation_days=365, eval_mode=eval_mode)
        self.eval_mode = eval_mode
        self.weather = weather
        self.action_type = action_type
        self.start_day = start_day
        self.start_month = start_month
        self.year = year
        self.daily_timestep = 0
        self.online = online
        self.action_buffer = []
        
        self.set_point = 289.15
        self.last_action = 0.0
        self.on = 0.0
        
        if action_type == "discrete":
            self.action_space = spaces.Discrete(11)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -20, 0]), 
                                            high=np.array([1, 1, 1, 20, 1]), 
                                            dtype=float)
        
    def set_set_point(self, set_point):
        self.set_point = set_point
                
    def _get_obs(self):
        outputs = self.env.get_output()
        delta = outputs["temRoo.T"] - self.set_point
        
        observation = np.array([
            normalize(outputs["heaPum.P"], 0, 5000),    # heat pump power
            normalize(outputs["temSup.T"], 273.15, 353.15),    # supply temperature
            normalize(outputs["TOut.T"], 253.15, 343.15),    # outdoor temperature
            delta,    # temperature error
            self.on
        ], dtype=float)
        
        return observation
    
    def hard_reset(self, year=2020, weather=None, options=None, seed=None):
        if weather is not None:
            self.weather = weather
        self.env = energym.make('SimpleHouseRad-v0', weather=self.weather, start_day=self.start_day, start_month=self.start_month, year=year, simulation_days=365, eval_mode=self.eval_mode)
        obs = self._get_obs()
        
        self.set_point = 289.15
        self.on = 0.0
        self.daily_timestep = 0

        return obs, {}
        
                
    def reset(self, options=None, seed=None):
        time.sleep(0.1)
        obs = self._get_obs()

        return obs, {}
    
    def step(self, action):
        if self.online:
            if self.daily_timestep == 84:
                self.set_point = 293.15
                self.on = 1.0
            elif self.daily_timestep == 252:
                self.set_point = 289.15
                self.on = 0.0
            elif self.daily_timestep == 288:
                self.daily_timestep = 0
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.action_type == "discrete":
            action = action * 0.1
        control = {'u': [action]}
        self.env.step(control)
        outputs = self._get_obs()
        reward = self.get_reward(outputs, action)
        self.last_action = action
        info = dict({
            "temRoo.T": self.env.get_output()["temRoo.T"]
        })
        
        if self.online:
            self.daily_timestep += 1

        return outputs, reward, False, False, info
    
    # reward 1
    # def get_reward(self, outputs, action):
    #     delta = outputs[3]
    #     heat_pump_power = outputs[0]
    #     if abs(delta) > 0.3:
    #         return -100
    #     else:
    #         smoothness_error = -2 * abs(action - self.last_action)
    #         energy_penalty = -6 * heat_pump_power
        
    #     reward = energy_penalty + smoothness_error
        
    #     return reward
    
    # reward 2
    def get_reward(self, outputs, action):
        delta = outputs[3]
        heat_pump_power = outputs[0]
        if abs(delta) > 0.3:
            return -100
        else:
            smoothness_error = -8 * abs(action - self.last_action)
            energy_penalty = -6 * heat_pump_power
        
        reward = energy_penalty + smoothness_error
        
        return reward