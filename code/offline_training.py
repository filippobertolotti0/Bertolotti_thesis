import d3rlpy
import registration
import numpy as np
from utils import get_dataset

if __name__ == "__main__":
    name = "ddpg_pretrained"
    
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
    
    if name == "bc_pretrained":
        learner = d3rlpy.algos.BCConfig().create(device="cuda:0")
    elif name == "ddpg_pretrained":
        learner = d3rlpy.algos.DDPGConfig(
                gamma=0.99,
                actor_learning_rate=0.001,
                critic_learning_rate=0.001,
                batch_size=512,
                tau=0.08
            ).create(device="cuda:0")
    
    learner.fit(dataset, n_steps=500000, n_steps_per_epoch=20000, show_progress=True, save_interval=20000)
    learner.save_model(f"./trained_models/{name}")