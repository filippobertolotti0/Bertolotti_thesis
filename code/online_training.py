import energym
import gymnasium as gym
import d3rlpy
import registration
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from utils import weathers
from imitation.data.types import Transitions
from imitation.algorithms import bc
from imitation.util.util import make_vec_env
from utils import get_dataset, unnormalize
import torch      

if __name__ == "__main__":
    env = gym.make("SwissHouseRSlaW2W-v0")
    name = "ddpg_only_online_new_parameters"
    d3rlpy.envs.seed_env(env, 42)
    
    # expert = d3rlpy.algos.BCConfig().create()
    # expert.build_with_env(env)
    # expert.load_model("./code/d3rlpy/trained_models/bc_pretrained")
    
    # expert = d3rlpy.algos.DDPGConfig().create()
    # expert.build_with_env(env)
    # expert.load_model("./code/d3rlpy/trained_models/ddpg_pretrained_new_hyperparameters")
    
    learner = d3rlpy.algos.DDPGConfig(
            gamma=0.99,
            actor_learning_rate=0.0002,
            critic_learning_rate=0.0002,
            batch_size=100,
            tau=0.02
        ).create()
    learner.build_with_env(env)
    # learner.copy_policy_from(expert)
    # learner.copy_q_function_from(expert)
    
    buffer = d3rlpy.dataset.FIFOBuffer(limit=100000)
    replay_buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=buffer,
        cache_size=100000,
        env=env,                                            
    )
    
    for epoch in range(6):
        print(f"Epoch: {epoch}")
        learner.fit_online(env=env, buffer=replay_buffer, n_steps=8064, n_steps_per_epoch=8064, save_interval=50000)
        learner.save_model(f"./code/d3rlpy/trained_models/{name}")
        
    out_list = []
    for episode in replay_buffer.episodes:
        for observation in episode.observations:
            out_list.append({
                "heaPum.P": unnormalize(observation[0], 0, 1000),
                "temSup.T": unnormalize(observation[1], 273.15, 353.15),
                "delta": observation[3]
            })
        
    out_df = pd.DataFrame(out_list)
    out_df.to_excel(f"./code/d3rlpy/datasets/{name}.xlsx")
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(12, 15))
    
    ax1.plot(out_df["delta"], color='b', label='Room Temperature')
    ax1.axhline(y=0, color='r', linestyle='--')
    
    ax2.plot(out_df["heaPum.P"], color='b', label='Heat Pump Power')

    plt.legend()
    plt.savefig(f"./code/d3rlpy/graphs/{name}.png")
    plt.show()