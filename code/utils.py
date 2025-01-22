import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN, DDPG, PPO, A2C, SAC, TD3
from tqdm import tqdm

DAY = 288
WEEK = DAY*7
MONTH = WEEK*4
HALF_YEAR = MONTH*6
YEAR = MONTH*12

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

def save_training_params(model_name, training_params):
    with open(f"./trained_models/{model_name}/training_params.txt", "w") as file:
        for key, value in training_params.items():
            file.write(f"{key} = {value}\n")
    
weathers = ["CH_BS_Basel", "CH_ZH_Maur", "CH_TI_Bellinzona", "CH_GR_Davos", "CH_GE_Geneva", "CH_VD_Lausanne"]

DDPG_PARAMS_ONLINE_RAPID_CONV = {
    'gamma': 0.99,
    'actor_learning_rate': 0.00004,
    'critic_learning_rate': 0.0003,
    'batch_size': 256,
    'tau': 0.05
}

DDPG_PARAMS_ONLINE_SLOW_CONV = {
    'gamma': 0.95,
    'actor_learning_rate': 0.0007,
    'critic_learning_rate': 0.0002,
    'batch_size': 2048,
    'tau': 0.05
}