import gymnasium as gym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import numpy as np
from tqdm import tqdm
from online_training import reward_plot, test
import registration
import matplotlib.pyplot as plt
import pandas as pd
from utils import unnormalize, save_training_params, weathers, DDPG_PARAMS_ONLINE, TD3_PARAMS_ONLINE, SAC_PARAMS_ONLINE, DAY, WEEK, MONTH, HALF_YEAR, YEAR
import torch
import os

if __name__ == "__main__":
    training_params = {
        "model_name": "",
        "expert_name": "",
        "episode_lenght": DAY,
        "agent_params_online": DDPG_PARAMS_ONLINE,
        "buffer_lenght": 50000,
        "seasons": [2023],
        "n_steps": MONTH*3,
        "update_interval": 2,
        "n_updates": 1
    }
    
    training_settings = {
        "save": True,
        "test": False,
        "rewards_graphs": True
    }
    
    env = gym.make("SimpleHouseRad-v0", start_day=1, start_month=10, episode_lenght=training_params["episode_lenght"],
                   training_schedule=True)
    name = training_params["model_name"]
    d3rlpy.envs.seed_env(env, 42)
    torch.cuda.manual_seed(42)
    out_list = []
    df = []
    
    expert = d3rlpy.algos.DDPGConfig().create()
    expert.build_with_env(env)
    expert.load_model(f"./trained_models/{training_params['expert_name']}/{training_params['expert_name']}")
    
    learner = d3rlpy.algos.DDPGConfig(
            tau=0.005,
            gamma=0.99,
            actor_learning_rate=0.0001,
            critic_learning_rate=0.0001,
            
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            # **training_params["agent_params_online"]
        ).create("cuda:0")
    learner.build_with_env(env)
    learner.copy_policy_from(expert)
    
    buffer = d3rlpy.dataset.FIFOBuffer(limit=training_params["buffer_lenght"])
    replay_buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=buffer,
        cache_size=200000,
        env=env,          
        action_space=d3rlpy.ActionSpace.CONTINUOUS,
        action_size=1                                  
    )
    
    for year in training_params["seasons"]:
        env.hard_reset(weather=f"aosta{year}")
        learner.fit_online(env=env, explorer=None, buffer=replay_buffer, n_steps=training_params["n_steps"], n_steps_per_epoch=training_params["episode_lenght"], save_interval=training_params["episode_lenght"],
                        show_progress=True, update_interval=training_params["update_interval"], n_updates=training_params["n_updates"], update_start_step=1, outputs=out_list)

    for observation, action, reward, temp_penalty, energy_penalty in out_list:
        # e_penalty = energy_penalty if observation[4] == 1 else 0
        df.append({
            "heaPum.P": unnormalize(observation[0], 0, 5000),
            "temSup.T": unnormalize(observation[1], 273.15, 353.15),
            "delta": observation[3],
            "reward": reward,
            "temp_penalty": temp_penalty,
            "energy_penalty": energy_penalty
            # "energy_penalty": e_penalty
        })
    out_df = pd.DataFrame(df)
    out_df["reward"] = pd.concat([pd.Series([0]), out_df["reward"][:-1]]).reset_index(drop=True)
    out_df = out_df.iloc[1:].reset_index(drop=True)
    
    env.close()
    
    #save model
    if training_settings["save"]:
        if not os.path.exists(f"./trained_models/{name}"):
            os.mkdir(f"./trained_models/{name}")
        learner.save_model(f"./trained_models/{name}/{name}")
        save_training_params(name, training_params)
        out_df.to_excel(f"./trained_models/{name}/{name}.xlsx")
    if training_settings["rewards_graphs"]:
        reward_plot(name, training_params["episode_lenght"])
    if training_settings["test"]:
        training_params["average temperature error"], training_params["average HeatPump power"] = test(learner, HALF_YEAR)
    
    print(f"-------- {name} training finished --------")