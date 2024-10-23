from matplotlib import pyplot as plt
import gymnasium as gym
import energym
import d3rlpy
import pandas as pd
import numpy as np
from tqdm import tqdm
import registration
from utils import unnormalize, weathers
from PID import PID

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
    
def reward_convergence(path, episode_days):
    df = pd.read_excel(path)["reward"]
    rewards = []
    step_number = []
    episode_steps = 288*episode_days
    r = len(df) // episode_steps
    
    for i in range(r):
        episode_reward = df[i*episode_steps:(i+1)*episode_steps]
        mean_reward = sum(episode_reward)/episode_steps
        if mean_reward > -100:
            rewards.append(mean_reward)
            step_number.append(i+1)
        
    plt.plot(step_number, rewards)
    plt.show()

if __name__ == "__main__":
    # training_comparison()
    # test_comparison()
    reward_convergence("./datasets/ddpg_only_online_3.xlsx", 2)