from matplotlib import pyplot as plt
import gymnasium as gym
import energym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import registration
from utils import unnormalize, DAY, WEEK, MONTH, HALF_YEAR, YEAR
from PID.PID import PID

def get_tick_positions(week_number: int):
    if week_number == 1:
        return range(0, 8064+1, 1152)
    elif week_number == 2:
        return range(8064, 16128+1, 1152)
    elif week_number == 3:
        return range(16128, 24192+1, 1152)
    elif week_number == 4:
        return range(24192, 32256+1, 1152)
    elif week_number == 5:
        return range(32256, 40320+1, 1152)
    elif week_number == 6:
        return range(40320, 48384+1, 1152)
    else:
        print("Invalid week number")
        
def get_week_data(df: pd.DataFrame, week_number: int):
    if week_number == 1:
        return df[:8064]
    elif week_number == 2:
        return df[8064:16128]
    elif week_number == 3:
        return df[16128:24192]
    elif week_number == 4:
        return df[24192:32256]
    elif week_number == 5:
        return df[32256:40320]
    elif week_number == 6:
        return df[40320:48384]
    else:
        print("Invalid week number")

def training_comparison():
    after_bc = pd.read_excel("./datasets/ddpg_after_bc.xlsx")
    after_training = pd.read_excel("./datasets/ddpg_after_pretraining.xlsx")
    only_online = pd.read_excel("./datasets/ddpg_only_online.xlsx")
    
    for week_number in range(1, 7):
        deltas_bc = get_week_data(after_bc['delta'], week_number)
        deltas_pre = get_week_data(after_training['delta'], week_number)
        deltas_online = get_week_data(only_online['delta'], week_number)
                
        tick_positions = get_tick_positions(week_number)
        tick_labels = tick_labels = [0, 4, 8, 12, 16, 20, 24, 28]
        
        plt.figure(figsize=(12, 6))
        plt.plot(deltas_bc, label='BC', color='b')
        plt.plot(deltas_pre, label='Pretraining', color='r')
        plt.plot(deltas_online, label='Online', color='g')
        plt.xticks(tick_positions, tick_labels)
        plt.ylim(-3, 2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        text_str = f"Week: {week_number}\nBC Mean: {abs(deltas_bc).mean():.2f}\nPretraining Mean: {abs(deltas_pre).mean():.2f}\nOnline Mean: {abs(deltas_online).mean():.2f}"
        plt.text(0.75, 0.05, text_str, transform=plt.gca().transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.savefig(f".\graphs\week_{week_number}.png")
    
    plt.show()
    
def test_comparison():
    out_list = []
    
    pid = PID()
    
    models = ["bc_pretrained_offline", "ddpg_pretrained_offline", "ddpg_from_scratch"]
    plt.figure(figsize=(12, 6))
    
    for model_name, color, label in zip(models, ["b", "r", "g"], ["After bc", "Pretrained offline", "Only online"]):
        out_list = []
        env = gym.make("SimpleHouseRad-v0", eval_mode=False)
        model = d3rlpy.algos.DDPGConfig().create()
        model.build_with_env(env)
        model.load_model(f"./trained_models/{model_name}")
        
        obs, _ = env.reset()
        steps = 288
        cumulative_error = 0
        
        env.set_set_point(289.15)
        
        for i in tqdm(range(steps)):
            if i == 84:
                env.set_set_point(293.15)
            if i == 252:
                env.set_set_point(289.15)
            action = model.predict(np.expand_dims(obs, axis=0))[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, rewards, terminated, truncated, info = env.step(action)
            cumulative_error += abs(obs[3])
            out_list.append({
                    "heaPum.P": unnormalize(obs[0], 0, 5000),
                    "temSup.T": unnormalize(obs[1], 273.15, 353.15),
                    "TOut.T": unnormalize(obs[2], 253.15, 343.15),
                    "temRoo.T": info['temRoo.T'],
                }
            )
            
        env.close()

        out_df = pd.DataFrame(out_list)
        print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
        print(f"Mean temperature error: {cumulative_error/steps}")
        
        plt.plot(out_df["temRoo.T"]-273.15, color=color, label=label)
        
    out_list = []
    env = energym.make("SimpleHouseRad-v0", simulation_days=365, eval_mode=False)
    outputs = env.get_output()
    set_point = 16
    cumulative_error = 0
    set_point = 16

    for i in tqdm(range(steps)):
        if i == 84:
            set_point = 20
        if i == 252:
            set_point = 16
        
        control_signal, cumulative_error = pid.predict(outputs, set_point, cumulative_error)
        
        control = {}
        control['u'] = [control_signal]
        outputs = env.step(control)
        out_list.append(outputs)
                
    out_df = pd.DataFrame(out_list)
    print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
    print(f"Mean temperature error: {cumulative_error/steps}")
    plt.plot(out_df["temRoo.T"]-273.15, color='k', label='PID')
    
    plt.ylabel('Room temperature')
    plt.xlabel('Steps (5 min)')
    plt.axhline(y=16, linestyle='--')
    plt.axhline(y=20, linestyle='--')   
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f"./graphs/test_comparison")
    plt.show()
    
def reward_convergence(path, episode_length):
    plt.figure(figsize=(12, 6)) 
    for p in path:
        df = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["reward"]
        rewards = []
        step_number = []
        # total_step = len(df)
        total_step = HALF_YEAR
        total_episodes = total_step // episode_length
        
        for i in range(total_episodes):
            episode_reward = df[i*episode_length:(i+1)*episode_length]
            # mean_reward = sum(episode_reward)/episode_length
            cumulative_reward = sum(episode_reward)
            rewards.append(cumulative_reward)
            step_number.append(i+1)
        plt.plot(step_number, rewards, label=p)
        
    for i in range(HALF_YEAR//episode_length, total_episodes, HALF_YEAR//episode_length):
        plt.axvline(x=i, color='g', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig("./graphs/reward_convergence/reward_convergence")
    plt.show()
    
def reward_convergence_splitted(path, episode_length):
    plt.figure(figsize=(12, 6)) 
    for p in path:
        temp_penalty = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["temp_penalty"]
        energy_penalty = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["energy_penalty"]
        temp_rewards = []
        energy_rewards = []
        step_number = []
        total_step = len(temp_penalty)
        total_episodes = total_step // episode_length
        
        for i in range(total_episodes):
            episode_temp_penalty = temp_penalty[i*episode_length:(i+1)*episode_length]
            episode_energy_penalty = energy_penalty[i*episode_length:(i+1)*episode_length]
            # mean_reward = sum(episode_reward)/episode_length
            temp_cumulative_reward = sum(episode_temp_penalty)
            energy_cumulative_reward = sum(episode_energy_penalty)
            temp_rewards.append(temp_cumulative_reward)
            energy_rewards.append(energy_cumulative_reward)
            step_number.append(i+1)
        plt.plot(step_number, temp_rewards, label=f"{p}: Temperature penalty")
        plt.plot(step_number, energy_rewards, label=f"{p}: Energy penalty")
        
    for i in range(HALF_YEAR//episode_length, total_episodes, HALF_YEAR//episode_length):
        plt.axvline(x=i, color='g', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.tight_layout()
    plt.savefig("./graphs/reward_convergence/reward_convergence_splitted")
    plt.legend()
    plt.show()
    
def test(model_name, test_lenght):
    # test parameters
    steps = test_lenght
    start_month = 10
    year = 2024
    low_temp = 16
    high_temp = 20
    turn_on = 7 * 12
    turn_off = 21 * 12
    
    env = gym.make("SimpleHouseRad-v0", start_month=start_month, year=year, start_day=1)
    d3rlpy.envs.seed_env(env, 42)
    f, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))
    
    # RL agent
    # expert = d3rlpy.algos.DDPGConfig(
    #         action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
    #     ).create()
    # expert.build_with_env(env)
    # expert.load_model(f"./trained_models/{model_name}/{model_name}")
    
    out_list = []
    obs, _ = env.reset()
    agent_cumulative_error = 0
    daily_timestep = 0
    env.set_set_point(low_temp)
    
    for i in tqdm(range(steps)):
        if daily_timestep == turn_on:
            env.set_set_point(high_temp)
        elif daily_timestep == turn_off:
            env.set_set_point(low_temp)
        elif daily_timestep == DAY:
            daily_timestep = 0
        action = model_name.predict(np.expand_dims(obs, axis=0))[0]
        obs, rewards, terminated, truncated, info = env.step(action)
        agent_cumulative_error += abs(obs[3])
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
    
    print(f"Mean HeatPump power: {mean_power_rl:.3f}")
    print(f"Mean temperature error: {mean_error_rl:.3f}")
    
    return mean_error_rl, mean_power_rl
    # ax1.plot(out_df["temRoo.T"]-273.15, color='b', label=model_name)
    # ax2.plot(out_df["heaPum.P"], color='b', label=model_name) 
    
    # PID agent
    # env = energym.make("SimpleHouseRad-v0", start_day=15, start_month=start_month, year=year, simulation_days=365, eval_mode=False)
    # pid = PID()
    
    # out_list = []
    # outputs = env.get_output()
    # daily_timestep = 0
    # set_point = low_temp

    # for i in tqdm(range(steps)):
    #     if daily_timestep == turn_on:
    #         set_point = high_temp
    #     elif daily_timestep == turn_off:
    #         set_point = low_temp
    #     elif daily_timestep == DAY:
    #         daily_timestep = 0
        
    #     control_signal = pid.predict(outputs, set_point)
        
    #     control = {}
    #     control['u'] = [control_signal]
    #     outputs = env.step(control)
    #     out_list.append(outputs)
        
    #     daily_timestep += 1
        
    # out_df = pd.DataFrame(out_list)
    
    # mean_power_pid = out_df['heaPum.P'].sum()/steps
    # mean_error_pid = pid.cumulative_error/steps
    
    # ax1.plot(out_df["temRoo.T"]-273.15, color='r', label='PID')
    # ax2.plot(out_df["heaPum.P"], color='r', label='PID')
    
    # for i in range(0, steps, DAY):
    #     ax1.plot([0+i, turn_on+i], [low_temp, low_temp], color='b', linestyle='--')
    #     ax1.plot([turn_on+i, turn_off+i], [high_temp, high_temp], color='b', linestyle='--')
    #     ax1.plot([turn_off+i, DAY+i], [low_temp, low_temp], color='b', linestyle='--')
    # for i in range(turn_on, steps, DAY):
    #     ax1.axvline(x=i, color='g', linestyle='--')
    # for i in range(turn_off, steps, DAY):
    #     ax1.axvline(x=i, color='g', linestyle='--')
    
    # ax1.set_ylabel('Room temperature')
    # ax2.set_ylabel('Heat pump power')
    # plt.xlabel('Steps (5 min)')
    
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f"./trained_models/{model_name}/{model_name}_test.png")
    
    # print("------------------------------------------------")
    # print(f"\tDDPG")
    # print(f"\tAverage HeatPump power: {mean_power_rl:.3f}")
    # print(f"\tAverage temperature error: {mean_error_rl:.3f}")
    # print("------------------------------------------------")
    # print(f"\tPID")
    # print(f"\tAverage HeatPump power: {mean_power_pid:.3f}")
    # print(f"\tAverage temperature error: {mean_error_pid:.3f}")
    # print("------------------------------------------------")
    
if __name__ == "__main__":
    # training_comparison()
    # test_comparison()
    reward_convergence(["ddpg_2days_reset_overlapped", "ddpg_2days_reset", "ddpg_no_FMU_reset"], DAY*2)
    # reward_convergence_splitted(["ddpg_continuative"], DAY*2)
    # test("ddpg_test_2", DAY*3)