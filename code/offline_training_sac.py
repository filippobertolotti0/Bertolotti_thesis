import os
import d3rlpy
import registration
from d3rlpy.preprocessing import MinMaxActionScaler
import numpy as np
from utils import get_dataset, save_training_params, SAC_PARAMS_OFFLINE

# def custom_epoch_callback(model: d3rlpy.algos.BC | d3rlpy.algos.DDPG, epoch: int, total_step: int):
#     epoch_steps = total_step/epoch
#     if total_step == 100000:
#         model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x100k")
#     elif total_step == 200000:
#         model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x200k")
#     elif total_step == 300000:
#         model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x300k")
#     elif total_step == 500000:
#         model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x500k")
#     elif total_step == 1000000:
#         model.save_model(f"./offline_training_gridsearch/{name}_{epoch_steps}x1000k")

if __name__ == "__main__":
    params = {
        "model_name": "sac_offline_2",
        "dataset": "PID_dataset_aosta.xlsx",
        "gamma": SAC_PARAMS_OFFLINE["gamma"],
        "actor_learning_rate": SAC_PARAMS_OFFLINE["actor_learning_rate"],
        "critic_learning_rate": SAC_PARAMS_OFFLINE["critic_learning_rate"],
        "temp_learning_rate": SAC_PARAMS_OFFLINE["temp_learning_rate"],
        "batch_size": SAC_PARAMS_OFFLINE["batch_size"],
        "tau": SAC_PARAMS_OFFLINE["tau"],
        "n_steps": SAC_PARAMS_OFFLINE["n_steps"],
        "n_steps_per_epoch": SAC_PARAMS_OFFLINE["n_steps_per_epoch"],
        "episode_lenght": 288
    }
    
    obs, next_obs, actions, rewards = get_dataset(f"./datasets/{params['dataset']}")
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
    
    learner = d3rlpy.algos.SACConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            gamma=params["gamma"],
            actor_learning_rate=params["actor_learning_rate"],
            critic_learning_rate=params["critic_learning_rate"],
            temp_learning_rate=params["temp_learning_rate"],
            batch_size=params["batch_size"],
            tau=params["tau"],
        ).create(device="cuda:0")
    
    learner.fit(dataset, n_steps=params["n_steps"], n_steps_per_epoch=params["n_steps_per_epoch"], show_progress=True, save_interval=params["n_steps"])
    if not os.path.exists(f"./trained_models/{params['model_name']}"):
        os.mkdir(f"./trained_models/{params['model_name']}")
    learner.save_model(f"./trained_models/{params['model_name']}/{params['model_name']}")
    save_training_params(params["model_name"], params)