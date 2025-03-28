from matplotlib import pyplot as plt
import gymnasium as gym
import energym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import registration
from utils import unnormalize, DAY, WEEK, MONTH, HALF_YEAR, YEAR, TD3_PARAMS_ONLINE, SAC_PARAMS_ONLINE
from PID.PID import PID

td3_params = {
    'gamma': 0.9,
    'actor_learning_rate': 0.0005965698722229415,
    'critic_learning_rate': 0.00013032913625000625,
    'batch_size': 2048,
    'tau': 0.05,
    'target_smoothing_sigma': 0.2,
    'target_smoothing_clip': 0.5,
    'update_actor_interval': 2,
}

sac_params = {
    'actor_learning_rate': 0.000993354483407106,
    'batch_size': 2048,
    'critic_learning_rate': 2.6679098378465916e-05,
    'gamma': 0.88,
    'n_critics': 4,
    'tau': 0.02,
    'temp_learning_rate': 0.0002015096624240672,
}

def get_tick_positions(week_number: int):
    if week_number == 1:
        return range(0, 8064+1, 1152)
    elif week_number == 2:
        return range(8064, 16128+1, 1152)
    elif week_number == 3:
        return range(16128, 24192+1, 1152)
    elif week_number == 4:
        return range(24192, 32256+1, 1152)
    elif week_number == 5:
        return range(32256, 40320+1, 1152)
    elif week_number == 6:
        return range(40320, 48384+1, 1152)
    else:
        print("Invalid week number")
        
def get_week_data(df: pd.DataFrame, week_number: int):
    if week_number == 1:
        return df[:8064]
    elif week_number == 2:
        return df[8064:16128]
    elif week_number == 3:
        return df[16128:24192]
    elif week_number == 4:
        return df[24192:32256]
    elif week_number == 5:
        return df[32256:40320]
    elif week_number == 6:
        return df[40320:48384]
    else:
        print("Invalid week number")
    
def reward_convergence(path, episode_length, total_steps):
    plt.figure(figsize=(12, 6)) 
    for p in path:
        df = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["reward"]
        rewards = []
        step_number = []
        # total_step = len(df)
        total_step = total_steps
        total_episodes = total_step // episode_length
        
        for i in range(total_episodes):
            episode_reward = df[i*episode_length:(i+1)*episode_length]
            # mean_reward = sum(episode_reward)/episode_length
            cumulative_reward = sum(episode_reward)
            rewards.append(cumulative_reward)
            step_number.append(i+1)
        plt.plot(step_number, rewards, label=p)
        
    # for i in range(HALF_YEAR//episode_length, total_episodes, HALF_YEAR//episode_length):
    #     plt.axvline(x=i, color='g', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig("./reward_convergence_off_on")
    plt.show()
    
def reward_convergence_splitted(path, episode_length):
    plt.figure(figsize=(12, 6)) 
    for p in path:
        temp_penalty = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["temp_penalty"]
        energy_penalty = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["energy_penalty"]
        temp_rewards = []
        energy_rewards = []
        step_number = []
        total_step = len(temp_penalty)
        total_episodes = total_step // episode_length
        
        for i in range(total_episodes):
            episode_temp_penalty = temp_penalty[i*episode_length:(i+1)*episode_length]
            episode_energy_penalty = energy_penalty[i*episode_length:(i+1)*episode_length]
            # mean_reward = sum(episode_reward)/episode_length
            temp_cumulative_reward = sum(episode_temp_penalty)
            energy_cumulative_reward = sum(episode_energy_penalty)
            temp_rewards.append(temp_cumulative_reward)
            energy_rewards.append(energy_cumulative_reward)
            step_number.append(i+1)
        plt.plot(step_number, temp_rewards, label=f"{p}: Temperature penalty")
        plt.plot(step_number, energy_rewards, label=f"{p}: Energy penalty")
        
    for i in range(HALF_YEAR//episode_length, total_episodes, HALF_YEAR//episode_length):
        plt.axvline(x=i, color='g', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.tight_layout()
    plt.savefig("./graphs/reward_convergence/reward_convergence_splitted")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    reward_convergence(["td3_off_on", "sac_off_on", "ddpg_off_on"], DAY, MONTH*2)
    # reward_convergence_splitted(["ddpg_continuative"], DAY*2)