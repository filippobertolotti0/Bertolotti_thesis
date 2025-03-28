import energym
import gymnasium as gym
from gymnasium import spaces
from utils import normalize, unnormalize, DAY, WEEK, MONTH, HALF_YEAR, YEAR
import numpy as np
import energym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

class SimpleHouseRad(gym.Env):    
    def __init__(self, weather="aosta2020", start_day=1, start_month=1, year=2010, episode_lenght=DAY, training_schedule=False):
        self.env = energym.make('SimpleHouseRad-v0', weather=weather, start_day=start_day, start_month=start_month, year=year, simulation_days=180, eval_mode=False)
        self.weather = weather
        self.start_day = start_day
        self.start_month = start_month
        self.year = year
        
        self.episode_lenght = episode_lenght
        self.timestep = 0
        self.daily_timestep = 0
        self.training_schedule = training_schedule
        self.schedule = 0
        self.set_point = 289.15
        
        self.weight_t = -2.4
        self.weight_e = -0.6
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
        
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, -30]), 
                                            high=np.array([1.0, 1.0, 1.0, 30]), 
                                            dtype=float)
        
        self.step(np.array([0.0]))
        
    def set_set_point(self, set_point):
        self.set_point = set_point + 273.15
                
    def _get_obs(self):
        outputs = self.env.get_output()
        delta = outputs["temRoo.T"] - self.set_point
        
        observation = np.array([
            normalize(outputs["heaPum.P"], 0, 5000),    # heat pump power
            normalize(outputs["temSup.T"], 273.15, 353.15),    # supply temperature
            normalize(outputs["TOut.T"], 253.15, 343.15),    # outdoor temperature
            delta,    # temperature error
        ], dtype=float)
        
        return observation
    
    def hard_reset(self, weather=None, options=None, seed=None):
        if weather is not None:
            self.weather = weather
            
        self.env.close()
        self.env = energym.make('SimpleHouseRad-v0', weather=self.weather, start_day=self.start_day, start_month=self.start_month, simulation_days=180)
        obs = self._get_obs()
        
        self.set_point = 289.15
        self.daily_timestep = 0
        self.timestep = 0

        return obs, {}
    
    def episode_reset(self, year=2020, month=10, day=1, weather=None, options=None, seed=None):
        if weather is not None:
            self.weather = weather
        self.env = energym.make('SimpleHouseRad-v0', weather=self.weather, start_day=day, start_month=month, year=year, simulation_days=2)
        obs = self._get_obs()
        
        self.set_point = 289.15
        self.daily_timestep = 0
        self.timestep = 0
        
        return obs, {}
                
    def reset(self, options=None, seed=None):
        obs = self._get_obs()

        return obs, {}
    
    def check_timestep(self):
        if self.training_schedule:
            if self.daily_timestep == 84:
                self.schedule = 1
                self.set_point = 293.15
            elif self.daily_timestep == 252:
                self.schedule = 0
                self.set_point = 289.15
            elif self.daily_timestep == 288:
                self.daily_timestep = 0
            
        self.daily_timestep += 1
        self.timestep += 1
            
    def step(self, action):
        self.check_timestep()
        outputs = self.env.get_output()
        if outputs["temRoo.T"] < 283.15:
            control = {'u': [1.0]}
            self.env.step(control)
            reward = -100
            outputs = self._get_obs()
        else:
            control = {'u': [action[0]]}
            self.env.step(control)
            outputs = self._get_obs()
            reward = self.get_reward(outputs)
        
        info = dict({
            "temRoo.T": self.env.get_output()["temRoo.T"]
        })
        
        if self.timestep == self.episode_lenght:
            terminated = True
            self.timestep = 0
        else:
            terminated = False

        return outputs, reward, terminated, False, info
    
    def get_reward(self, outputs):        
        delta = abs(outputs[3])
        heat_pump_power = outputs[0]
            
        temperature_penalty = self.weight_t * delta
        energy_penalty = self.weight_e * heat_pump_power
        
        reward = temperature_penalty + energy_penalty

        return reward