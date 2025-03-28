from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import registration
import gymnasium
import optuna
from utils import DAY, WEEK, MONTH, HALF_YEAR, YEAR, get_dataset, normalize, unnormalize
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_dashboard import run_server
from data_comparison import test
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import traceback

N_TRIALS = 1000
N_STARTUP_TRIALS = 40
N_EVALUATIONS = 3

ENV_ID = "SimpleHouseRad-v0"

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

def test(model, test_lenght):
    # test parameters
    steps = test_lenght
    start_month = 10
    year = 2024
    low_temp = 16
    high_temp = 20
    turn_on = 7 * 12
    turn_off = 21 * 12
    
    env = gymnasium.make("SimpleHouseRad-v0", start_month=start_month, year=year, start_day=1)
    d3rlpy.envs.seed_env(env, 42)
    
    # out_list = []
    rewards = []
    obs, _ = env.reset()
    # agent_cumulative_error = 0
    daily_timestep = 0
    env.set_set_point(low_temp)
    
    for i in tqdm(range(steps)):
        if daily_timestep == turn_on:
            env.set_set_point(high_temp)
        elif daily_timestep == turn_off:
            env.set_set_point(low_temp)
        elif daily_timestep == DAY:
            daily_timestep = 0
        action = model.predict(np.expand_dims(obs, axis=0))[0]
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        daily_timestep += 1
        
    env.close()
    
    return sum(rewards) / len(rewards)

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    actor_lr = trial.suggest_loguniform('actor_learning_rate', 1e-5, 1e-3)
    critic_lr = trial.suggest_loguniform('critic_learning_rate', 1e-5, 1e-3)
    temp_lr = trial.suggest_loguniform('temp_learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    n_steps = trial.suggest_categorical("n_steps", [50000, 100000, 200000, 300000, 400000, 500000])
    n_steps_per_epoch = trial.suggest_categorical("n_steps_per_epoch", [5000, 10000, 20000, 40000, 50000, 100000])
    n_critics = trial.suggest_categorical("n_critics", [1, 2, 3, 4])

    hyperparams = {
        "gamma": 0.88,
        "tau": tau,
        "actor_learning_rate": actor_lr,
        "critic_learning_rate": critic_lr,
        "temp_learning_rate": temp_lr,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_steps_per_epoch": n_steps_per_epoch,
        "n_critics": n_critics
    }
    
    return hyperparams

def objective(trial: optuna.Trial) -> float:
    params = sample_params(trial)
    model_hyperparams = {
        "gamma": params["gamma"],
        "tau": params["tau"],
        "actor_learning_rate": params["actor_learning_rate"],
        "critic_learning_rate": params["critic_learning_rate"],
        "batch_size": params["batch_size"],
        "temp_learning_rate": params["temp_learning_rate"],
        "n_critics": params["n_critics"],
    }

    model = d3rlpy.algos.SACConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            **model_hyperparams
        ).create("cuda:0")

    model.fit(dataset=dataset, n_steps=params["n_steps"], n_steps_per_epoch=params["n_steps_per_epoch"], save_interval=params["n_steps"], show_progress=True)
        
    average_reward = test(model, HALF_YEAR)
    print(f"average_reward: {average_reward}")

    return average_reward

if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS)
    storage = "sqlite:///offline_study.db"
    
    study = optuna.create_study(study_name="offline_study_sac_real", sampler=sampler, pruner=pruner, storage=storage,  direction="maximize", load_if_exists=True)
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    except Exception as e:
        print("An error occurred during optimization.")
        traceback.print_exc()

    print("Number of finished trials: ", len(study.trials))
    print("last month evaluated")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    run_server(storage)