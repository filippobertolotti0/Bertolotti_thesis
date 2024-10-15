import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN, DDPG, PPO, A2C, SAC, TD3
from tqdm import tqdm

def normalize(x, min, max):
    return (x - min) / (max - min)

def unnormalize(x, min, max):
    return x * (max - min) + min

def get_dataset(path):
    df = pd.read_excel(path)
        
    obs = []
        
    for _, row in tqdm(df.iterrows()):
        row['heaPum.P'] = normalize(row['heaPum.P'], 0, 5000)
        row['temSup.T'] = normalize(row['temSup.T'], 273.15, 353.15)
        row['TOut.T'] = normalize(row['TOut.T'], 253.15, 343.15)
        row['delta'] = row['delta']
        
        obs.append([row['heaPum.P'], row['temSup.T'], row['TOut.T'], row['delta']])
    
    next_obs = obs[1:len(obs)]
    obs.pop()
        
    acts = df['acts'].values
    rewards = df['rewards'].values
    
    return obs, next_obs, acts, rewards

class CustomCallback(BaseCallback):
    def __init__(self, out_list, vec_env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.out_list = out_list
        self.env = vec_env

    def _on_step(self):
        self.out_list.append({
                "heaPum.P": unnormalize(self.locals['new_obs'][0,0], 0, 1000),
                "temSup.T": unnormalize(self.locals['new_obs'][0,1], 273.15, 353.15),
                # "temRoo.T": unnormalize(self.locals['new_obs']['temRoo_T'][0,0], 263.15, 343.15),
                "temRoo.T": self.locals['infos'][0]['temRoo.T'],
                # "temSup.T": unnormalize(self.locals['new_obs']['temSup_T'][0,0], 273.15, 353.15),
                # "TOut.T": unnormalize(self.locals['new_obs']['TOut_T'][0,0], 253.15, 343.15),
                # "heaPum.P": unnormalize(self.locals['new_obs']['heaPum_P'][0,0], 0, 1000),
                "delta": self.locals['new_obs'][0,3]
            }
        )
        
        return True
    
weathers = ["CH_BS_Basel", "CH_ZH_Maur", "CH_TI_Bellinzona", "CH_GR_Davos", "CH_GE_Geneva", "CH_VD_Lausanne"]

algorithms = {
    "DDPG": {"alg": DDPG,
        "alg_name": "DDPG",
        "params": {"gamma": 0.99, 'learning_rate': 0.001, 'batch_size': 512, 'buffer_size': 100000, 'tau': 0.08, 'train_freq': 1, "policy_kwargs": dict(net_arch=[64, 64])},
        "action_type": "continuous"
    },
    "SAC": {"alg": SAC,
        "alg_name": "SAC",
        "params": {},
        "action_type": "continuous"
    },
    "TD3": {"alg": TD3,
        "alg_name": "TD3",
        "params": {},
        "action_type": "continuous"
    },
    "A2C": {"alg": A2C,
        "alg_name": "A2C",
        "params": {},
        "action_type": "continuous"
    },
    "PPO": {"alg": PPO,
        "alg_name": "PPO",
        "params": {},
        "action_type": "continuous"
    },
    "DQN": {"alg": DQN,
        "alg_name": "DQN",
        "params": {},
        "action_type": "discrete"
    },
}