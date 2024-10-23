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

if __name__ == "__main__":
    env = gym.make("SimpleHouseRad-v0", eval_mode=False, online=False)
    pid = PID()
    d3rlpy.envs.seed_env(env, 42)
    model_name = ""
    
    expert = d3rlpy.algos.DDPGConfig().create()
    expert.build_with_env(env)
    expert.load_model("./trained_models/ddpg_only_online_3")
    
    model = d3rlpy.algos.DDPGConfig(
            gamma=0.99,
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            batch_size=512,
            tau=0.08
        ).create()
    model.build_with_env(env)
    model.copy_policy_from(expert)
    # model.copy_q_function_from(expert)
    # model.save_model("./trained_models/ddpg_from_scratch")
    
    out_list = []
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))
    
    obs, _ = env.reset()
    steps = 288
    cumulative_error = 0
    
    env.set_set_point(289.15)
    daily_timestep = 0
    
    for i in tqdm(range(steps)):
        if daily_timestep == 84:
            env.set_set_point(293.15)
        elif daily_timestep == 252:
            env.set_set_point(289.15)
        elif daily_timestep == 288:
            daily_timestep = 0
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
        
        daily_timestep += 1
        
    env.close()

    out_df = pd.DataFrame(out_list)
    print(f"DDPG\n---------------------------------------")
    print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
    print(f"Mean temperature error: {cumulative_error/steps}")
    
    ax1.plot(out_df["temRoo.T"]-273.15, color='b', label=model_name)
    ax2.plot(out_df["heaPum.P"], color='b', label=model_name)
        
    out_list = []
    env = energym.make("SimpleHouseRad-v0", simulation_days=365, eval_mode=False)
    outputs = env.get_output()
    
    cumulative_error = 0
    daly_timestep = 0
    set_point = 16

    for i in tqdm(range(steps)):
        if daily_timestep == 84:
                set_point = 20
        elif daily_timestep == 252:
            set_point = 16
        elif daily_timestep == 288:
            daily_timestep = 0
        
        control_signal, cumulative_error = pid.predict(outputs, set_point, cumulative_error)
        
        control = {}
        control['u'] = [control_signal]
        outputs = env.step(control)
        out_list.append(outputs)
        
        daily_timestep += 1
        
    out_df = pd.DataFrame(out_list)
    print(f"PID\n---------------------------------------")
    print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
    print(f"Mean temperature error: {cumulative_error/steps}")
    
    ax1.plot(out_df["temRoo.T"]-273.15, color='r', label='PID')
    ax2.plot(out_df["heaPum.P"], color='r', label='PID')
    
    for i in range(0, steps, 288):
        ax1.plot([0+i, 84+i], [16, 16], color='b', linestyle='--')
        ax1.plot([84+i, 252+i], [20, 20], color='b', linestyle='--')
        ax1.plot([252+i, 288+i], [16, 16], color='b', linestyle='--')
    for i in range(84, steps, 288):
        ax1.axvline(x=i, color='g', linestyle='--')
    for i in range(252, steps, 288):
        ax1.axvline(x=i, color='g', linestyle='--')
    
    ax1.set_ylabel('Room temperature')
    ax2.set_ylabel('Heat pump power')
    plt.xlabel('Steps (5 min)')
      
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f"./graphs/test_comparison")
    plt.show()