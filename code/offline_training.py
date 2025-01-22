import os
import d3rlpy
import registration
from d3rlpy.preprocessing import MinMaxActionScaler
import numpy as np
from utils import get_dataset

def custom_epoch_callback(model: d3rlpy.algos.BC | d3rlpy.algos.DDPG, epoch: int, total_step: int):
    epoch_steps = total_step/epoch
    if total_step == 100000:
        model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x100k")
    elif total_step == 200000:
        model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x200k")
    elif total_step == 300000:
        model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x300k")
    elif total_step == 500000:
        model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x500k")
    elif total_step == 1000000:
        model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x1000k")

if __name__ == "__main__":
    name = "best_offline"
    
    params = {
        "gamma": 0.88,
        "actor_learning_rate": 0.0000934293426,
        "critic_learning_rate": 0.0008577842194694033,
        "batch_size": 256,
        "tau": 0.05,
        "n_steps": 400000,
        "n_steps_per_epoch": 50000,
        "episode_lenght": 288
    }
    
    obs, next_obs, actions, rewards = get_dataset("./datasets/PID_dataset_basel.xlsx")
    obs = np.array(obs)
    next_obs = np.array(next_obs)
    rewards = np.array(rewards)

    dones = []
    for i in range(len(obs)):
        dones.append(False)
    dones[-1] = True
    dones = np.array(dones)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=dones
    )
    
    learner = d3rlpy.algos.DDPGConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            gamma=params["gamma"],
            actor_learning_rate=params["actor_learning_rate"],
            critic_learning_rate=params["critic_learning_rate"],
            batch_size=params["batch_size"],
            tau=params["tau"]
        ).create(device="cuda:0")
    
    learner.fit(dataset, n_steps=params["n_steps"], n_steps_per_epoch=params["n_steps_per_epoch"], show_progress=True, save_interval=params["n_steps"])
    if not os.path.exists(f"./trained_models/{name}"):
        os.mkdir(f"./trained_models/{name}")
    learner.save_model(f"./trained_models/{name}/{name}")