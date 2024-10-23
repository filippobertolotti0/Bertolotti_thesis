import energym
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from utils import weathers
    
if __name__ == "__main__":
    env = energym.make("SimpleHouseRad-v0", simulation_days=365, eval_mode=False)
    
    steps = 288
    last_error = 0
    total_error = 0
    cumulative_error = 0
    set_point = 16
    
    kp = 0.1
    ki = 0
    kd = 250
    
    out_list = []
    outputs = env.get_output()
    
    lower_bound = 500
    upper_bound = 1000

    for i in tqdm(range(steps)):
        if i == 96:
            set_point = 20
        #     lower_bound = 1250
        #     upper_bound = 2000
        # if i == 2000:
        #     set_point = 19
        #     lower_bound = 2250
        #     upper_bound = 3000
        if i == 252:
            set_point = 16
        #     lower_bound = 3250
        #     upper_bound = 4000
        # is_between = lower_bound < i < upper_bound
        # time = (i + 1) * 300
        error = (outputs['temRoo.T'] - 273.15) - set_point
        total_error += (-error)
        cumulative_error += abs(error)
        delta_error = (-error) - last_error
        heat_P_power = outputs['heaPum.P']/5000
        
        control_signal = kp * (-error) + ki * 300 * total_error + (kd/300) * delta_error
        heat_P_power += control_signal
        control_signal = max(0, min(1, heat_P_power))
        
        control = {}
        control['u'] = [control_signal]
        outputs = env.step(control)
        out_list.append(outputs)
        
        last_error = -error
        print(f"{outputs['heaPum.COP']}")
        
    out_df = pd.DataFrame(out_list)
    print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
    print(f"Mean temperature error: {cumulative_error/steps}")
    
    f, (ax1,ax2) = plt.subplots(2,figsize=(10,15))

    ax1.plot(out_df['temRoo.T']-273.15, 'r')
    ax1.axhline(y=16, color='r', linestyle='--')
    # ax1.axhline(y=19, color='r', linestyle='--')
    ax1.axhline(y=20, color='r', linestyle='--')
    # ax1.axhline(y=22, color='r', linestyle='--')
    ax1.set_ylabel('Temp')
    ax1.set_xlabel('Steps')

    ax2.plot(out_df['heaPum.P'], 'g')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Steps')

    plt.subplots_adjust(hspace=0.4)
    # plt.savefig("./graphs/PID")
    plt.show()