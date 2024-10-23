import gymnasium as gym
import d3rlpy
import registration
import matplotlib.pyplot as plt
import pandas as pd
from utils import unnormalize
import torch


DAY = 288
WEEK = DAY*7
MONTH = WEEK*4
HALF_YEAR = MONTH*6
YEAR = MONTH*12

if __name__ == "__main__":
    env = gym.make("SimpleHouseRad-v0", start_day=1, start_month=10, year=2014, online=True)
    name = "ddpg_only_online_3"
    d3rlpy.envs.seed_env(env, 42)
    torch.cuda.manual_seed(42)
    out_list = []
    episode_lenght = HALF_YEAR
    
    # expert = d3rlpy.algos.BCConfig().create()
    # expert.build_with_env(env)
    # expert.load_model("./trained_models/bc_pretrained_offline")
    
    # expert = d3rlpy.algos.DDPGConfig().create()
    # expert.build_with_env(env)
    # expert.load_model("./trained_models/ddpg_pretrained_offline")
    
    learner = d3rlpy.algos.DDPGConfig(
            gamma=0.99,
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            batch_size=1024,
            tau=0.01
        ).create()
    learner.build_with_env(env)
    # learner.copy_policy_from(expert)
    # learner.copy_q_function_from(expert)
    
    for year in [2015, 2016, 2017, 2018]:
        env.hard_reset(year=year)
        
        buffer = d3rlpy.dataset.FIFOBuffer(limit=50000)
        replay_buffer = d3rlpy.dataset.ReplayBuffer(
            buffer=buffer,
            cache_size=50000,
            env=env,                                            
        )
        
        for episode in range((HALF_YEAR)//episode_lenght):
            print(f"YEAR: {year} - Episode: {episode+1}/{(HALF_YEAR)//episode_lenght}")
            learner.fit_online(env=env, buffer=replay_buffer, n_steps=episode_lenght, n_steps_per_epoch=episode_lenght, save_interval=episode_lenght, show_progress=True, update_interval=DAY)
            
        for episode in replay_buffer.episodes:
            for observation, reward in zip(episode.observations, episode.rewards):
                out_list.append({
                    "heaPum.P": unnormalize(observation[0], 0, 5000),
                    "temSup.T": unnormalize(observation[1], 273.15, 353.15),
                    "delta": observation[3],
                    "reward": reward[0],
                    # "on": observation[4]
                })
        learner.save_model(f"./trained_models/{name}")
            
    env.close()
        
    out_df = pd.DataFrame(out_list)
    out_df.to_excel(f"./datasets/{name}.xlsx")
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(12, 15))
    
    ax1.plot(out_df["delta"], color='b', label='Room Temperature')
    ax1.axhline(y=0, color='r', linestyle='--')
    
    ax2.plot(out_df["heaPum.P"], color='b', label='Heat Pump Power')

    plt.legend()
    plt.savefig(f"./graphs/{name}.png")
    plt.show()