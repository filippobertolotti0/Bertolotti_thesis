import energym
import registration
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import weathers
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from utils import normalize
import matplotlib.pyplot as plt 

def reward_function(obs, action, last_action):
    delta = outputs['delta']
    heat_pump_power = normalize(outputs['heaPum.P'], 0, 1000)
    
    if abs(delta) < 1:
        smoothness_error = -2 * abs(action - last_action)
    else:
        smoothness_error = 0
    temp_error = -4 * abs(delta)
    energy_penalty = -3 * heat_pump_power
    reward = temp_error + energy_penalty + smoothness_error
    
    return reward
    
if __name__ == "__main__":
    env = energym.make("SimpleHouseRad-v0", weather=weathers[0], simulation_days=365, start_day = 1, start_month = 10, year = 2020, eval_mode=False)
    
    steps = 288
    last_error = 0
    total_error = 0
    cumulative_error = 0
    set_point = 16
    last_action = 0
        
    kp = 0.1
    ki = 0
    kd = 100
    
    outputs = env.get_output()
    outputs['delta'] = (outputs['temRoo.T'] - 273.15) - set_point
    out_list = []
    out_list.append(outputs)
    
    acts = []
    tem_room = []
    rewards = []
    set_points = []
    
    daily_timestep = 0

    for i in tqdm(range(steps*2)):
        if daily_timestep == 84:
            set_point = 20
        elif daily_timestep == 252:
            set_point = 16
        elif daily_timestep == 288:
            daily_timestep = 0
        time = (i + 1) * 300
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
        tem_room.append(outputs['temRoo.T'] - 273.15)
        out_list.append(outputs)
        acts.append([control_signal])
        rewards.append(reward_function(outputs, control_signal, last_action))
        
        last_error = -error
        last_action = control_signal
        daily_timestep += 1
        set_points.append(set_point)
        
    out_df = pd.DataFrame(out_list)
    out_df = out_df[["heaPum.P", "temSup.T", "TOut.T", "delta"]]
    obs = out_df[:steps].values
    next_obs = out_df[1:(steps+1)].values
    acts.append([0])
    rewards.append(0)
    
    dataset = pd.DataFrame({
        'heaPum.P': out_df['heaPum.P'],
        'temSup.T': out_df['temSup.T'],
        'TOut.T': out_df['TOut.T'],
        'delta': out_df['delta'],
        'acts': np.array(acts).flatten(),
        'rewards': rewards,
    })
    
    dataset.to_excel('datasets/PID_dataset_basel.xlsx')
    
    # plt.plot(out_df["TOut.T"])
    plt.plot(set_points, color='r', linestyle='--')
    plt.show()