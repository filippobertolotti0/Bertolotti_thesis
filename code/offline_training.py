import d3rlpy
import registration
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

if __name__ == "__main__":
    name = "bc_pretrained_offline"
    
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
    
    if name == "bc_pretrained_offline":
        learner = d3rlpy.algos.BCConfig(
                batch_size=512,
                learning_rate=0.001
            ).create(device="cuda:0")
    elif name == "ddpg_pretrained_offline":
        learner = d3rlpy.algos.DDPGConfig(
                gamma=0.99,
                actor_learning_rate=0.001,
                critic_learning_rate=0.001,
                batch_size=512,
                tau=0.08
            ).create(device="cuda:0")
    
    learner.fit(dataset, n_steps=300000, n_steps_per_epoch=10000, show_progress=True, save_interval=10000)
    learner.save_model(f"./trained_models/{name}")