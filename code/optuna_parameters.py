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
    # gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.88, 0.9, 0.95, 0.99])
    # actor_lr = trial.suggest_loguniform('actor_learning_rate', 1e-5, 1e-3)
    # critic_lr = trial.suggest_loguniform('critic_learning_rate', 1e-5, 1e-3)
    # batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    # buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5*1e4), int(1e5), int(2*1e5)])
    # tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # episode_lenght = trial.suggest_categorical("episode_lenght", [DAY, DAY*2, WEEK, MONTH, MONTH*3, HALF_YEAR])
    # update_interval = trial.suggest_categorical("update_interval", [1, 2, 4, DAY//32, DAY//16, DAY//8, DAY//4, DAY//2, DAY, DAY*2])
    # n_updates = trial.suggest_categorical("n_updates", [1, 2, 3, 4, 5])
    weight_t = trial.suggest_float("weight_t", -5, 0, step=0.01)
    weight_e = trial.suggest_float("weight_e", -5, 0, step=0.1)

    hyperparams = {
        "weight_t": weight_t,
        "weight_e": weight_e
        # "gamma": gamma,
        # "tau": tau,
        # "actor_learning_rate": actor_lr,
        # "critic_learning_rate": critic_lr,
        # "batch_size": batch_size,
        # "buffer_size": buffer_size,
        # "episode_lenght": episode_lenght,
        # "update_interval": update_interval,
        # "n_updates": n_updates
    }
    
    return hyperparams

def objective(trial: optuna.Trial) -> float:
    params = sample_params(trial)
    # model_hyperparams = {
    #     "gamma": params["gamma"],
    #     "tau": params["tau"],
    #     "actor_learning_rate": params["actor_learning_rate"],
    #     "critic_learning_rate": params["critic_learning_rate"],
    #     "batch_size": params["batch_size"],
    # }

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
        
    error, power = test(model, DAY*28)
    
    if error > 0.8:
        return -10
    
    evaluation = -(normalize(error, 0, 0.8) + normalize(power, 0, 1300))
    print(f"Error: {error}, Power: {power}, Evaluation: {evaluation}")

    return evaluation

if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS)
    storage = "sqlite:///weight_study_3.db"
    
    study = optuna.create_study(study_name="weight_study_3", sampler=sampler, pruner=pruner, storage=storage,  direction="maximize", load_if_exists=True)
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
    
    """
    RAPID CONVERGENCE
    gamma: 0.99
    actor_learning_rate: 4.419700505347808e-05
    critic_learning_rate: 0.000260378074201075
    batch_size: 256
    buffer_size: 100000
    tau: 0.05
    episode_lenght: 2016
    update_interval: 1
    n_updates: 2
    """
    
    """
    BEST CONVERGENCE AFTER 2 SEASONS
    gamma: 0.95
    actor_learning_rate: 0.0007262202266591029
    critic_learning_rate: 0.00021011289833049228
    batch_size: 2048
    buffer_size: 50000
    tau: 0.05
    episode_lenght: 288
    update_interval: 2
    n_updates: 3
    """
    
    """
    weight_t: -3.77
    weight_e: -1.3
    """