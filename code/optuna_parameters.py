from typing import Any
from typing import Dict

import registration
import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_dashboard import run_server
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

N_TRIALS = 1000
N_STARTUP_TRIALS = 100
N_EVALUATIONS = 3
N_TIMESTEPS = int(10000)

ENV_ID = "SwissHouseRSlaW2W-v0"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.99, 0.999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
    }
    
    return hyperparams

class FlattenDictWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return gymnasium.spaces.flatten(self.env.observation_space, observation)


class RewardAccumulatorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardAccumulatorCallback, self).__init__(verbose)
        self.total_reward = 0

    def reset(self):
        self.total_reward = 0

    def _on_step(self) -> bool:
        self.total_reward += self.locals['rewards'][0]
        return True

def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sample_params(trial))

    env = Monitor(gymnasium.make(ENV_ID))
    env = FlattenDictWrapper(env)
    model = DDPG(env=env, **kwargs)

    reward_accumulator = RewardAccumulatorCallback()

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=reward_accumulator)
    except AssertionError as e:
        print(e)
        nan_encountered = True

    if nan_encountered:
        return float("nan")

    episode_rewards = reward_accumulator.total_reward
    mean_reward = episode_rewards / N_TIMESTEPS if episode_rewards else float("nan")
    model.env.close()

    return mean_reward


if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS)
    storage = optuna.storages.InMemoryStorage()
    
    study = optuna.create_study(sampler=sampler, pruner=pruner, storage=storage,  direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=4)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    run_server(storage)