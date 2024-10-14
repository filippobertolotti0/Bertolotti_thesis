import energym
import gymnasium as gym
from gymnasium import spaces
from utils import normalize, unnormalize
import numpy as np

import energym
import gymnasium as gym
from gymnasium import spaces
from utils import normalize, unnormalize
import numpy as np

class SimpleHouseRad(gym.Env):    
    def __init__(self, action_type="continuous", weather="CH_BS_Basel", eval_mode=False):
        self.env = energym.make('SimpleHouseRad-v0', weather=weather, simulation_days=365, eval_mode=eval_mode)
        self.eval_mode = eval_mode
        self.weather = weather
        self.action_type = action_type
        self.start_day = 1
        self.start_month = 1
        self.on = False
        
        self.total_temp_error = 0
        self.set_point = 289.15
        self.last_action = 0
        
        if action_type == "discrete":
            self.action_space = spaces.Discrete(11)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -20]), 
                                            high=np.array([1, 1, 1, 20]), 
                                            dtype=float)
        
    def set_set_point(self, set_point):
        self.set_point = set_point
        
    def set_random_start_day(self):
        start_day = np.random.randint(1, 29)
        start_month = np.random.choice([1, 2, 3, 9, 10, 11, 12]) 
        self.start_day = start_day
        self.start_month = start_month
                
    def _get_obs(self):
        outputs = self.env.get_output()
        delta = outputs["temRoo.T"] - self.set_point
        
        observation = np.array([
            normalize(outputs["heaPum.P"], 0, 5000),    # heat pump power
            normalize(outputs["temSup.T"], 273.15, 353.15),    # supply temperature
            normalize(outputs["TOut.T"], 253.15, 343.15),    # outdoor temperature
            delta    # temperature delta
        ], dtype=float)
        
        return observation
                
    def reset(self, weather=None, options=None, seed=None):
        if weather is not None:
            self.weather = weather
        self.env = energym.make('SwissHouseRSlaW2W-v0', weather=self.weather, start_day=self.start_day, start_month=self.start_month, simulation_days=365, eval_mode=self.eval_mode)
        obs = self._get_obs()

        return obs, {}
    
    def step(self, action):
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

        return outputs, reward, False, False, info
    
    def get_reward(self, outputs, action):
        delta = outputs[3]  # Delta è l'ultimo valore nell'osservazione
        heat_pump_power = outputs[0]  # Il primo valore è heat pump power
        
        if abs(delta) < 1:
            smoothness_error = -2 * abs(action - self.last_action)
        else:
            smoothness_error = 0
        temp_error = -4 * abs(delta)
        energy_penalty = -3 * heat_pump_power
        reward = temp_error + energy_penalty + smoothness_error
        
        return reward

# class swissHouseRSlaW2W(gym.Env):    
#     def __init__(self, action_type="continuous", weather="CH_BS_Basel", eval_mode=False):
#         self.env = energym.make('SwissHouseRSlaW2W-v0', weather=weather, simulation_days=365, eval_mode=eval_mode)
#         self.eval_mode = eval_mode
#         self.weather = weather
#         self.action_type = action_type
#         self.start_day = 1
#         self.start_month = 1
        
#         self.total_temp_error = 0
#         self.set_point = 289.15
#         self.last_action = 0
        
#         if action_type == "discrete":
#             self.action_space = spaces.Discrete(11)
#         else:
#             self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        
#         self.observation_space = spaces.Dict(
#             {
#                 "heaPum_P": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
#                 "temSup_T": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
#                 "TOut_T": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
#                 "delta": spaces.Box(low=-20, high=20, shape=(1,), dtype=float)
#             }
#         )
        
#     def set_set_point(self, set_point):
#         self.set_point = set_point
        
#     def set_random_start_day(self):
#         start_day = np.random.randint(1, 29)
#         # start_month = np.random.randint(1, 13)
#         start_month = np.random.choice([1, 2, 3, 9, 10, 11, 12]) 
#         self.start_day = start_day
#         self.start_month = start_month
                
#     def _get_obs(self):
#         outputs = self.env.get_output()
#         delta = outputs["temRoo.T"] - self.set_point 
        
#         return {
#             "heaPum_P": np.array([normalize(outputs["heaPum.P"], 0, 1000)], dtype=float),    #heat pump power
#             "temSup_T": np.array([normalize(outputs["temSup.T"], 273.15, 353.15)], dtype=float),    #supply temperature
#             "TOut_T": np.array([normalize(outputs["TOut.T"], 253.15, 343.15)], dtype=float),        #outdoor temperature
#             "delta": np.array([delta], dtype=float)       #temperature delta
#         }
                
#     def reset(self, weather=None, options=None, seed=None):
#         if weather != None:
#             self.weather = weather
#         self.env = energym.make('SwissHouseRSlaW2W-v0', weather=self.weather, start_day=self.start_day, start_month=self.start_month, simulation_days=365, eval_mode=self.eval_mode)
#         obs = self._get_obs()

#         return obs, {}
    
#     def step(self, action):
#         if self.action_type == "discrete":
#             action = action * 0.1
#         control = {'u': [action]}
#         self.env.step(control)
#         outputs = self._get_obs()
#         reward = self.get_reward(outputs, action)[0]
#         self.last_action = action
#         info = dict({
#             "temRoo.T": self.env.get_output()["temRoo.T"]
#         })

#         return outputs, reward, False, False, info
    
#     def get_reward(self, outputs, action):
#         delta = outputs["delta"]
#         heat_pump_power = outputs["heaPum_P"]
        
#         if abs(delta) < 1:
#             smoothness_error = -2 * abs(action - self.last_action)
#         else:
#             smoothness_error = 0
#         temp_error = -4 * abs(delta)
#         energy_penalty = -3 * heat_pump_power
#         reward = temp_error + energy_penalty + smoothness_error
        
#         return reward
