import gymnasium as gym
import d3rlpy
import registration
import numpy as np
from utils import get_dataset

if __name__ == "__main__":
    env = gym.make("SimpleHouseRad-v0")

    obs, next_obs, actions, rewards = get_dataset("./datasets/PID_dataset.xlsx")
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
            gamma=0.99,
            actor_learning_rate=0.0002,
            critic_learning_rate=0.0002,
            batch_size=100,
            tau=0.02
        ).create(device="cuda:0")
    
    learner.fit(dataset, n_steps=300000, n_steps_per_epoch=20000, show_progress=True, save_interval=20000)
    learner.save_model("./code/d3rlpy/trained_models/ddpg_pretrained_new_hyperparameters")