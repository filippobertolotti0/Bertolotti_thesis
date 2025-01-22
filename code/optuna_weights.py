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
N_STARTUP_TRIALS = 150
N_EVALUATIONS = 3
N_TIMESTEPS = HALF_YEAR

ENV_ID = "SimpleHouseRad-v0"

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    weight_t = trial.suggest_float("weight_t", -2.7, -2.2, step=0.05)
    weight_e = trial.suggest_float("weight_e", -0.8, -0.4, step=0.05)

    hyperparams = {
        "weight_t": weight_t,
        "weight_e": weight_e
    }
    
    return hyperparams

def objective(trial: optuna.Trial) -> float:
    params = sample_params(trial)

    env = gymnasium.make(ENV_ID, start_day=1, start_month=10, year=2015, episode_lenght=DAY, training_schedule=True, weight_t=params["weight_t"], weight_e=params["weight_e"])  
    model = d3rlpy.algos.DDPGConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            gamma=0.95,
            actor_learning_rate=0.0007,
            critic_learning_rate=0.0002,
            batch_size=2048,
            tau=0.05
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
    for year in [2015, 2016, 2017]:
        if year != 2015: env.hard_reset(year=year)
        model.fit_online(env=env, buffer=replay_buffer, n_steps=N_TIMESTEPS, n_steps_per_epoch=DAY, save_interval=N_TIMESTEPS,
                            update_interval=DAY//8, n_updates=2, show_progress=True, outputs=out_list)
    for _, _, reward in out_list:
        rewards.append(reward)
        
    error, power = test(model, HALF_YEAR)
    
    evaluation = -(normalize(error, 0, 0.55) + normalize(power, 0, 1100))

    return evaluation

if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=110, n_warmup_steps=MONTH*3)
    storage = "sqlite:///weight_study.db"
    
    study = optuna.create_study(study_name="weight_study", sampler=sampler, pruner=pruner, storage=storage,  direction="maximize", load_if_exists=True)
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