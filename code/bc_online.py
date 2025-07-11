import gymnasium as gym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import numpy as np
from tqdm import tqdm
import registration
import matplotlib.pyplot as plt
import pandas as pd
from utils import unnormalize, save_training_params, weathers, DDPG_PARAMS_ONLINE_SLOW_CONV, DAY, WEEK, MONTH, HALF_YEAR, YEAR, TD3_PARAMS_ONLINE, SAC_PARAMS_ONLINE
import torch
import os

def my_callback(model: d3rlpy.algos.DDPG, epoch: int, total_step: int):
    print(f"step: {total_step}")
    
def reward_plot(model_name, episode_length):
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    df = pd.read_excel(f"./trained_models/{model_name}/{model_name}.xlsx")
    rewards = df["reward"]
    temp_penalty = df["temp_penalty"]
    energy_penalty = df["energy_penalty"]

    cumulative_rewards = []
    temp_rewards = []
    energy_rewards = []
    step_number = []

    total_steps = len(rewards)
    total_episodes = total_steps // episode_length

    for i in range(total_episodes):
        episode_rewards = rewards[i * episode_length:(i + 1) * episode_length]
        episode_temp_penalty = temp_penalty[i * episode_length:(i + 1) * episode_length]
        episode_energy_penalty = energy_penalty[i * episode_length:(i + 1) * episode_length]

        cumulative_rewards.append(sum(episode_rewards))
        temp_rewards.append(sum(episode_temp_penalty))
        energy_rewards.append(sum(episode_energy_penalty))
        step_number.append(i + 1)

    ax1.plot(step_number, cumulative_rewards, label=model_name)
    ax2.plot(step_number, temp_rewards, label=f"Temperature penalty")
    ax2.plot(step_number, energy_rewards, label=f"Energy penalty")

    for i in range(HALF_YEAR // episode_length, total_episodes, HALF_YEAR // episode_length):
        ax1.axvline(x=i, color='g', linestyle='--')
        ax2.axvline(x=i, color='g', linestyle='--')

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative reward")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"./trained_models/{model_name}/reward_convergence.png")

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative reward")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"./trained_models/{model_name}/reward_convergence_splitted.png")

if __name__ == "__main__":
    training_params = {
        "model_name": "bc_td3",
        "episode_lenght": DAY,
        "agent_params": TD3_PARAMS_ONLINE,
        "buffer_lenght": 50000,
        "seasons": [2020],
        "n_steps": MONTH*2,
        "update_interval": 2,
        "n_updates": 1
    }
    
    training_settings = {
        "save": True,
        "test": False,
        "rewards_graphs": False
    }
    
    env = gym.make("SimpleHouseRad-v0", start_day=1, start_month=10, episode_lenght=training_params["episode_lenght"], training_schedule=True)
    name = training_params["model_name"]
    d3rlpy.envs.seed_env(env, 42)
    torch.cuda.manual_seed(42)
    out_list = []
    df = []
    
    expert = d3rlpy.algos.BCConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
        ).create()
    expert.build_with_env(env)
    expert.load_model(f"./trained_models/bc/bc")
    
    learner = d3rlpy.algos.TD3Config(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            **training_params["agent_params"]
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