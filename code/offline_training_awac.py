import os
import d3rlpy
import registration
from d3rlpy.preprocessing import MinMaxActionScaler
import numpy as np
from utils import get_dataset, save_training_params, TD3_PARAMS_OFFLINE

if __name__ == "__main__":
    params = {
        "model_name": "",
        "dataset": "",
        "n_steps": 200000,
        "n_steps_per_epoch": 20000,
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
    
    learner = d3rlpy.algos.CQLConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
        ).create(device="cuda:0")
    
    learner.fit(dataset, n_steps=params["n_steps"], n_steps_per_epoch=params["n_steps_per_epoch"], show_progress=True, save_interval=params["n_steps"])
    if not os.path.exists(f"./trained_models/{params['model_name']}"):
        os.mkdir(f"./trained_models/{params['model_name']}")
    learner.save_model(f"./trained_models/{params['model_name']}/{params['model_name']}")
    save_training_params(params["model_name"], params)