from typing import Any
from typing import Dict

import registration
import gymnasium
import optuna
from utils import DAY, WEEK, MONTH, HALF_YEAR, YEAR, normalize
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_dashboard import run_server
from data_comparison import test
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import traceback

N_TRIALS = 1000
N_STARTUP_TRIALS = 100
N_EVALUATIONS = 3
N_TIMESTEPS = HALF_YEAR

ENV_ID = "SimpleHouseRad-v0"

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = trial.suggest_categorical("gamma", [0.88, 0.9, 0.95, 0.99])
    actor_lr = trial.suggest_loguniform('actor_learning_rate', 1e-5, 1e-3)
    critic_lr = trial.suggest_loguniform('critic_learning_rate', 1e-5, 1e-3)
    temp_lr = trial.suggest_loguniform('temp_learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    tau = trial.suggest_categorical("tau", [0.01, 0.02, 0.05, 0.08])
    update_interval = trial.suggest_categorical("update_interval", [4, DAY//32, DAY//16, DAY//8, DAY//4, DAY//2, DAY, DAY*2])
    n_updates = trial.suggest_categorical("n_updates", [1, 2, 3])
    n_critics = trial.suggest_categorical("n_critics", [1, 2, 3, 4])

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "actor_learning_rate": actor_lr,
        "critic_learning_rate": critic_lr,
        "temp_learning_rate": temp_lr,
        "batch_size": batch_size,
        "update_interval": update_interval,
        "n_updates": n_updates,
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

    env = gymnasium.make(ENV_ID, weather="aosta2020", start_day=1, start_month=10, year=2020, episode_lenght=DAY, training_schedule=True)  
    model = d3rlpy.algos.SACConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            **model_hyperparams
        ).create("cuda:0")
    model.build_with_env(env)
    
    buffer = d3rlpy.dataset.FIFOBuffer(limit=50000)
    replay_buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=buffer,
        cache_size=200000,
        env=env,          
        action_space=d3rlpy.ActionSpace.CONTINUOUS,
        action_size=1                                  
    )

    out_list = []
    rewards = []
    for year, weather in zip([2020, 2021], ["aosta2020", "aosta2021"]):
        if year != 2020: env.hard_reset(weather=weather)
        model.fit_online(env=env, buffer=replay_buffer, n_steps=N_TIMESTEPS, n_steps_per_epoch=DAY, save_interval=N_TIMESTEPS,
                            update_interval=params["update_interval"], n_updates=params["n_updates"], show_progress=True, outputs=out_list)
    for _, _, reward, _, _ in out_list:
        rewards.append(reward)
        
    error, power = test(model, HALF_YEAR)
    
    if error > 0.6 or power > 1760:
        return -10
    
    evaluation = -(normalize(error, 0, 0.6) + normalize(power, 0, 1750))
    print(f"Error: {error}, Power: {power}, Evaluation: {evaluation}")

    return evaluation

if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS)
    storage = "sqlite:///TD3_study.db"
    
    study = optuna.create_study(study_name="SAC_study", sampler=sampler, pruner=pruner, storage=storage,  direction="maximize", load_if_exists=True)
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
    # {'actor_learning_rate': 0.000993354483407106, 'batch_size': 2048, 'critic_learning_rate': 2.6679098378465916e-05, 'gamma': 0.88, 'n_critics': 4, 'n_updates': 1, 'tau': 0.02, 'temp_learning_rate': 0.0002015096624240672, 'update_interval': 4}