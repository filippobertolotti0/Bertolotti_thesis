import pandas as pd
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

TD3_PARAMS_ONLINE = {
    'gamma': 0.9,
    'actor_learning_rate': 0.0005965698722229415,
    'critic_learning_rate': 0.00013032913625000625,
    'batch_size': 2048,
    'tau': 0.05,
    'target_smoothing_sigma': 0.2,
    'target_smoothing_clip': 0.5,
    'update_actor_interval': 2,
    'update_interval': 9,
    'n_updates': 2
}

TD3_PARAMS_OFFLINE = {
    'actor_learning_rate': 0.0003899037287948072,
    'critic_learning_rate': 0.00023468554130439816,
    'batch_size': 128,
    'tau': 0.001,
    'gamma': 0.88,
    'target_smoothing_sigma': 0.5,
    'target_smoothing_clip': 0.3,
    'update_actor_interval': 4,
    'n_steps': 400000,
    'n_steps_per_epoch': 100000
}

SAC_PARAMS_ONLINE = {
    'actor_learning_rate': 0.000993354483407106,
    'batch_size': 2048,
    'critic_learning_rate': 2.6679098378465916e-05,
    'gamma': 0.88,
    'n_critics': 4,
    'tau': 0.02,
    'temp_learning_rate': 0.0002015096624240672,
    'update_interval': 8,
    'n_updates': 1,
}

SAC_PARAMS_OFFLINE = {
    'actor_learning_rate': 0.00013252793344665855,
    'critic_learning_rate': 1.190008670186549e-05,
    'temp_learning_rate': 1.1222270343816039e-05,
    'gamma': 0.88,
    'batch_size': 512,
    'tau': 0.05,
    'n_steps': 50000,
    'n_steps_per_epoch': 20000,
    'n_critics': 2
}