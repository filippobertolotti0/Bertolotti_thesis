from matplotlib import pyplot as plt
import gymnasium as gym
import energym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import registration
from utils import weathers, unnormalize, normalize, DAY, WEEK, MONTH, HALF_YEAR, YEAR
from PID.PID import PID

if __name__ == "__main__":
    # test parameters
    steps = HALF_YEAR
    start_day = 1
    start_month = 10
    year = 2024
    low_temp = 16
    high_temp = 20
    turn_on = 7 * 12
    turn_off = 21 * 12
    weather = "aosta2019"
    schedule = 0
    
    env = gym.make("SimpleHouseRad-v0", weather=weather, start_month=start_month, year=year, start_day=start_day)
    d3rlpy.envs.seed_env(env, 42)
    f, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))
    
    # RL agent
    model_name = "aosta_1"
    
    model = d3rlpy.algos.DDPGConfig(
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
        ).create()
    model.build_with_env(env)
    model.load_model(f"./trained_models/{model_name}/{model_name}")
    # model.copy_policy_from(expert)
    
    out_list = []
    rewards_list = []
    obs, _ = env.reset()
    agent_cumulative_error = 0
    daily_timestep = 0
    env.set_set_point(low_temp)
    
    for i in tqdm(range(steps)):
        if daily_timestep == turn_on:
            env.set_set_point(high_temp)
            schedule = 1
        elif daily_timestep == turn_off:
            env.set_set_point(low_temp)
            schedule = 0
        elif daily_timestep == DAY:
            daily_timestep = 0
        action = model.predict(np.expand_dims(obs, axis=0))[0]
        obs, rewards, terminated, truncated, info = env.step(action)
        agent_cumulative_error += abs(obs[3])
        rewards_list.append(rewards)
        out_list.append({
                "heaPum.P": unnormalize(obs[0], 0, 5000),
                "temSup.T": unnormalize(obs[1], 273.15, 353.15),
                "TOut.T": unnormalize(obs[2], 253.15, 343.15),
                "temRoo.T": info['temRoo.T'],
            }
        )
        
        daily_timestep += 1
        
    env.close()

    out_df = pd.DataFrame(out_list)
    mean_power_rl = out_df['heaPum.P'].sum()/steps
    mean_error_rl = agent_cumulative_error/steps
    ax1.plot(out_df["temRoo.T"]-273.15, color='b', label=model_name)
    ax2.plot(out_df["heaPum.P"], color='b', label=model_name) 
    
    # PID agent
    env = energym.make("SimpleHouseRad-v0", weather=weather, start_day=start_day, start_month=start_month, year=year, simulation_days=365, eval_mode=False)
    pid = PID()
    
    out_list = []
    outputs = env.get_output()
    daily_timestep = 0
    set_point = low_temp

    for i in tqdm(range(steps)):
    # for i in tqdm(range(steps)):
        if daily_timestep == turn_on:
            set_point = high_temp
        elif daily_timestep == turn_off:
            set_point = low_temp
        elif daily_timestep == DAY:
            daily_timestep = 0
        
        control_signal = pid.predict(outputs, set_point)
        
        control = {}
        control['u'] = [control_signal]
        outputs = env.step(control)
        out_list.append(outputs)
        
        daily_timestep += 1
        
    out_df = pd.DataFrame(out_list)
    
    mean_power_pid = out_df['heaPum.P'].sum()/steps
    mean_error_pid = pid.cumulative_error/steps
    
    ax1.plot(out_df["temRoo.T"]-273.15, color='r', label='PID')
    ax2.plot(out_df["heaPum.P"], color='r', label='PID')
    
    for i in range(0, steps, DAY):
        ax1.plot([0+i, turn_on+i], [low_temp, low_temp], color='b', linestyle='--')
        ax1.plot([turn_on+i, turn_off+i], [high_temp, high_temp], color='b', linestyle='--')
        ax1.plot([turn_off+i, DAY+i], [low_temp, low_temp], color='b', linestyle='--')
    for i in range(turn_on, steps, DAY):
        ax1.axvline(x=i, color='g', linestyle='--')
    for i in range(turn_off, steps, DAY):
        ax1.axvline(x=i, color='g', linestyle='--')
    
    ax1.set_ylabel('Room temperature')
    ax2.set_ylabel('Heat pump power')
    plt.xlabel('Steps (5 min)')
    
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f"./09_01/jan_test_off")
    
    print("------------------------------------------------")
    print(f"\tDDPG")
    print(f"\tAverage HeatPump power: {mean_power_rl:.3f}")
    print(f"\tAverage temperature error: {mean_error_rl:.3f}")
    print("------------------------------------------------")
    print(f"\tPID")
    print(f"\tAverage HeatPump power: {mean_power_pid:.3f}")
    print(f"\tAverage temperature error: {mean_error_pid:.3f}")
    print("------------------------------------------------")
    print(f"\tSaving: {(mean_power_pid-mean_power_rl)/(mean_power_pid/100):.3f}%")
    print(f"\tAverage reward: {sum(rewards_list)/len(rewards_list)}")
    
    plt.show()