import energym
from utils import normalize
import registration
import numpy as np
import pandas as pd
from tqdm import tqdm

def reward_function(obs):
    delta = abs(obs['delta'])
    heat_pump_power = normalize(obs['heaPum.P'], 0, 5000)

    weight_t = -2.4
    weight_e = -0.6
        
    temperature_error = weight_t * delta
    energy_penalty = weight_e * heat_pump_power     
    
    reward = temperature_error + energy_penalty
    
    return reward
    
if __name__ == "__main__":
    steps = 50000
    
    kp = 0.1
    ki = 0
    kd = 220
    
    out_list = []
    acts = []
    rewards = []
    set_points = []
    
    for year in [2020, 2021, 2022, 2023]:
        env = energym.make("SimpleHouseRad-v0", weather=f"aosta{year}", simulation_days=190, start_day=1, start_month=10, year=year, eval_mode=False)
        
        last_error = 0
        total_error = 0
        cumulative_error = 0
        set_point = 16
        last_action = 0
        
        outputs = env.get_output()
        outputs['delta'] = (outputs['temRoo.T'] - 273.15) - set_point
        outputs['on'] = 0.0
        out_list.append(outputs)
        rewards.append(0)
        
        daily_timestep = 0

        for i in tqdm(range(steps)):
            if daily_timestep == 84:
                set_point = 20
            elif daily_timestep == 252:
                set_point = 16
            elif daily_timestep == 288:
                daily_timestep = 0
            error = (outputs['temRoo.T'] - 273.15) - set_point
            total_error += (-error)
            delta_error = (-error) - last_error
            heat_P_power = outputs['heaPum.P']/5000
            
            control_signal = kp * (-error) + ki * 300 * total_error + (kd/300) * delta_error
            heat_P_power += control_signal
            control_signal = max(0, min(1, heat_P_power))
            
            control = {}
            control['u'] = [control_signal]
            outputs = env.step(control)
            outputs['delta'] = (outputs['temRoo.T'] - 273.15) - set_point
            out_list.append(outputs)
            acts.append([control_signal])
            rewards.append(reward_function(outputs))
            
            last_error = -error
            last_action = control_signal
            daily_timestep += 1
            set_points.append(set_point)
        
        acts.append([0])
        
    out_df = pd.DataFrame(out_list)
    out_df = out_df[["heaPum.P", "temSup.T", "TOut.T", "delta"]]
    
    dataset = pd.DataFrame({
        'heaPum.P': out_df['heaPum.P'],
        'temSup.T': out_df['temSup.T'],
        'TOut.T': out_df['TOut.T'],
        'delta': out_df['delta'],
        'acts': np.array(acts).flatten(),
        'rewards': rewards,
    })
    
    dataset.to_excel('datasets/PID_dataset_aosta.xlsx')