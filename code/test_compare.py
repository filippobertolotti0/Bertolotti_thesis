from matplotlib import pyplot as plt
import gymnasium as gym
import energym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import registration
from utils import unnormalize, DAY, WEEK, MONTH, HALF_YEAR, YEAR, TD3_PARAMS_ONLINE, SAC_PARAMS_ONLINE
from PID.PID import PID

td3_params = {
    'gamma': 0.9,
    'actor_learning_rate': 0.0005965698722229415,
    'critic_learning_rate': 0.00013032913625000625,
    'batch_size': 2048,
    'tau': 0.05,
    'target_smoothing_sigma': 0.2,
    'target_smoothing_clip': 0.5,
    'update_actor_interval': 2,
}

sac_params = {
    'actor_learning_rate': 0.000993354483407106,
    'batch_size': 2048,
    'critic_learning_rate': 2.6679098378465916e-05,
    'gamma': 0.88,
    'n_critics': 4,
    'tau': 0.02,
    'temp_learning_rate': 0.0002015096624240672,
}

def unnormalize(value, min_val, max_val):
    """Utility function to unnormalize values"""
    return value * (max_val - min_val) + min_val

def run_simulation(model=None, is_pid=False, steps=HALF_YEAR, start_day=1, start_month=10, 
                   year=2024, low_temp=16, high_temp=20, turn_on=7*12, turn_off=21*12, 
                   weather="aosta2019"):
    # Consistent environment setup
    if is_pid:
        env = energym.make(
            "SimpleHouseRad-v0", 
            weather=weather, 
            start_day=start_day, 
            start_month=start_month, 
            year=year, 
            simulation_days=365, 
            eval_mode=False
        )
        pid = PID()
        outputs = env.get_output()
        set_point = low_temp
        daily_timestep = 0
        pid_cumulative_error = 0
        out_list = []

        for i in tqdm(range(steps)):
            if daily_timestep == turn_on:
                set_point = high_temp
            elif daily_timestep == turn_off:
                set_point = low_temp
            
            # Consistent daily timestep reset
            if daily_timestep >= DAY:
                daily_timestep = 0
            
            control_signal = pid.predict(outputs, set_point)
            control = {'u': [control_signal]}
            outputs = env.step(control)
            out_list.append(outputs)
            pid_cumulative_error += abs(outputs['temRoo.T'] - set_point - 273.15)
            
            daily_timestep += 1
        
        out_df = pd.DataFrame(out_list)
        mean_power = out_df['heaPum.P'].sum() / steps
        mean_error = pid_cumulative_error / steps

        return {
            "out_df": out_df,
            "mean_power": mean_power,
            "mean_error": mean_error,
            "rewards_list": []
        }
    
    else:
        # RL Model Simulation
        env = gym.make(
            "SimpleHouseRad-v0", 
            weather=weather, 
            start_month=start_month, 
            year=year, 
            start_day=start_day
        )
        d3rlpy.envs.seed_env(env, 42)
        
        out_list = []
        rewards_list = []
        obs, _ = env.reset()
        agent_cumulative_error = 0
        daily_timestep = 0
        env.set_set_point(low_temp)
        
        for i in tqdm(range(steps)):
            if daily_timestep == turn_on:
                env.set_set_point(high_temp)
            elif daily_timestep == turn_off:
                env.set_set_point(low_temp)
            
            # Consistent daily timestep reset
            if daily_timestep >= DAY:
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
            })
            
            daily_timestep += 1
        
        env.close()
        out_df = pd.DataFrame(out_list)
        
        mean_power = out_df['heaPum.P'].sum() / steps
        mean_error = agent_cumulative_error / steps
        mean_reward = sum(rewards_list) / len(rewards_list)
        
        return {
            "out_df": out_df,
            "mean_power": mean_power,
            "mean_error": mean_error,
            "mean_reward": mean_reward,
            "rewards_list": rewards_list
        }

def main():
    # Simulation parameters
    steps = HALF_YEAR
    start_day = 1
    start_month = 10
    year = 2024
    low_temp = 16
    high_temp = 20
    turn_on = 7 * 12
    turn_off = 21 * 12
    weather = "aosta2019"
    
    # Create figure
    f, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))
    
    # Models to compare
    models_info = [
        {
            "name": "bc_1y_20_21", 
            "config": d3rlpy.algos.BCConfig,
            "color": "b",
            "label": "bc_ddpg",
            "params": {}
        },
        {
            "name": "bc_1y_20_22", 
            "config": d3rlpy.algos.BCConfig,
            "color": "b",
            "label": "bc_ddpg",
            "params": {}
        },
        {
            "name": "bc_1y_21_22", 
            "config": d3rlpy.algos.BCConfig,
            "color": "b",
            "label": "bc_ddpg",
            "params": {}
        },
        {
            "name": "bc_1y_21_23", 
            "config": d3rlpy.algos.BCConfig,
            "color": "b",
            "label": "bc_ddpg",
            "params": {}
        },
        {
            "name": "bc_1y_22_23", 
            "config": d3rlpy.algos.BCConfig,
            "color": "b",
            "label": "bc_ddpg",
            "params": {}
        },
        
        # {
        #     "name": "bc_td3", 
        #     "config": d3rlpy.algos.TD3Config,
        #     "color": "r",
        #     "label": "bc_td3",
        #     "params": td3_params
        # },
        # {
        #     "name": "cql_sac", 
        #     "config": d3rlpy.algos.SACConfig,
        #     "color": "g",
        #     "label": "cql_sac",
        #     "params": sac_params
        # },
        
        # {
        #     "name": "td3_off_on_3m_20", 
        #     "config": d3rlpy.algos.TD3Config,
        #     "color": "b",
        #     "label": "td3_1m_20",
        #     "params": td3_params
        # },
        # {
        #     "name": "td3_off_on_3m_21", 
        #     "config": d3rlpy.algos.TD3Config,
        #     "color": "r",
        #     "label": "td3_1m_21",
        #     "params": td3_params
        # },
        # {
        #     "name": "td3_off_on_3m_22", 
        #     "config": d3rlpy.algos.TD3Config,
        #     "color": "b",
        #     "label": "td3_1m_22",
        #     "params": td3_params
        # },
        # {
        #     "name": "td3_off_on_3m_23", 
        #     "config": d3rlpy.algos.TD3Config,
        #     "color": "r",
        #     "label": "td3_1m_23",
        #     "params": td3_params
        # },
        # {
        #     "name": "td3_off_on_3m_24", 
        #     "config": d3rlpy.algos.TD3Config,
        #     "color": "b",
        #     "label": "td3_1m_24",
        #     "params": td3_params
        # },
        
        # {
        #     "name": "sac_off_on_3m_20", 
        #     "config": d3rlpy.algos.SACConfig,
        #     "color": "m",
        #     "label": "sac_1m_20",
        #     "params": {}
        # },
        # {
        #     "name": "sac_off_on_3m_21", 
        #     "config": d3rlpy.algos.SACConfig,
        #     "color": "m",
        #     "label": "sac_1m_21",
        #     "params": {}
        # },
        # {
        #     "name": "sac_off_on_3m_22", 
        #     "config": d3rlpy.algos.SACConfig,
        #     "color": "m",
        #     "label": "sac_1m_22",
        #     "params": {}
        # },
        # {
        #     "name": "sac_off_on_3m_23",
        #     "config": d3rlpy.algos.SACConfig,
        #     "color": "m",
        #     "label": "sac_1m_23",
        #     "params": {}
        # },
        # {
        #     "name": "sac_off_on_3m_24",
        #     "config": d3rlpy.algos.SACConfig,
        #     "color": "m",
        #     "label": "sac_1m_24",
        #     "params": {}
        # },        
        
        # {
        #     "name": "ddpg_off_on_3m_20", 
        #     "config": d3rlpy.algos.DDPGConfig,
        #     "color": "r",
        #     "label": "ddpg_1m_20",
        #     "params": {}
        # },
        # {
        #     "name": "ddpg_off_on_3m_21", 
        #     "config": d3rlpy.algos.DDPGConfig,
        #     "color": "b",
        #     "label": "ddpg_1m_21",
        #     "params": {}
        # },
        # {
        #     "name": "ddpg_off_on_3m_22", 
        #     "config": d3rlpy.algos.DDPGConfig,
        #     "color": "k",
        #     "label": "ddpg_1m_22",
        #     "params": {}
        # },
        # {
        #     "name": "ddpg_off_on_3m_23", 
        #     "config": d3rlpy.algos.DDPGConfig,
        #     "color": "m",
        #     "label": "ddpg_1m_23",
        #     "params": {}
        # },
        # {
        #     "name": "ddpg_off_on_3m_24", 
        #     "config": d3rlpy.algos.DDPGConfig,
        #     "color": "g",
        #     "label": "ddpg_1m_24",
        #     "params": {}
        # },
    ]
    
    # Results storage
    results = []
    
    # Run simulations for each model
    for model_info in models_info:
        # Create and load model
        model = model_info['config'](
            action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
            **model_info.get('params', {})
        ).create()
        
        # Build environment and load model
        env = gym.make("SimpleHouseRad-v0")
        model.build_with_env(env)
        model.load_model(f"./trained_models/{model_info['name']}/{model_info['name']}")
        
        # Run simulation
        rl_results = run_simulation(
            steps=steps,
            model=model, 
            start_day=start_day, 
            start_month=start_month, 
            year=year, 
            low_temp=low_temp, 
            high_temp=high_temp, 
            turn_on=turn_on, 
            turn_off=turn_off, 
            weather=weather
        )
        
        # Plot results
        ax1.plot(rl_results['out_df']["temRoo.T"] - 273.15, color=model_info['color'], label=model_info['name'])
        ax2.plot(rl_results['out_df']["heaPum.P"], color=model_info['color'], label=model_info['name'])
        
        # Store results
        results.append({
            "model": model_info['name'],
            "power": rl_results['mean_power'],
            "error": rl_results['mean_error'],
            "reward": rl_results['mean_reward']
        })
    
    # Run PID simulation
    # pid_results = run_simulation(
    #     steps=steps,
    #     model=model, 
    #     start_day=start_day, 
    #     start_month=start_month, 
    #     year=year, 
    #     low_temp=low_temp, 
    #     high_temp=high_temp, 
    #     turn_on=turn_on, 
    #     turn_off=turn_off, 
    #     weather=weather,
    #     is_pid=True
    # )
    
    # # Plot PID results
    # ax1.plot(pid_results['out_df']["temRoo.T"] - 273.15, color='k', label='PID')
    # ax2.plot(pid_results['out_df']["heaPum.P"], color='k', label='PID')
    
    # # Add results for PID
    # results.append({
    #     "model": "PID",
    #     "power": pid_results['mean_power'],
    #     "error": pid_results['mean_error'],
    #     "reward": 0
    # })
    
    # Plotting details
    ax1.set_ylabel('Room temperature (°C)')
    ax2.set_ylabel('Heat pump power (W)')
    plt.xlabel('Steps (5 min)')
    
    # Add temperature threshold lines
    ax1.axhline(y=low_temp, color='grey', linestyle='--')
    ax1.axhline(y=high_temp, color='grey', linestyle='--')
    
    ax1.axvline(x=turn_on, color='y', linestyle='--')
    ax1.axvline(x=turn_off, color='y', linestyle='--')
    ax2.axvline(x=turn_on, color='y', linestyle='--')
    ax2.axvline(x=turn_off, color='y', linestyle='--')
    
    plt.tight_layout()
    plt.legend()
    
    # Print results
    print("Comparative Results:")
    for result in results:
        print(f"Model: {result['model']}")
        print(f"Mean Power: {result['power']:.3f}")
        print(f"Mean Error: {result['error']:.3f}")
        print(f"Mean Reward: {result['reward']:.3f}")
        print(f"Saving: {(results[-1]['power'] - result['power'])/(results[-1]['power']/100):.3f}%")
        print("------------------------------------------------")
    
    # plt.savefig("il_rl.png")
    plt.show()

if __name__ == "__main__":
    main()