import os
import d3rlpy
import registration
from d3rlpy.preprocessing import MinMaxActionScaler
import numpy as np
from utils import get_dataset, save_training_params, SAC_PARAMS_OFFLINE

if __name__ == "__main__":
    params = {
        "model_name": "sac_off_1y_22_23",
        "dataset": "PID_dataset_aosta_22_23.xlsx",
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