import energym
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from utils import weathers
import time
    
if __name__ == "__main__":
    env = energym.make("SimpleHouseRad-v0", simulation_days=365, eval_mode=False)
    
    steps = 288*2
    last_error = 0
    total_error = 0
    cumulative_error = 0
    set_point = 20
    
    kp = 0.1
    ki = 0
    kd = 250
    
    out_list = []
    outputs = env.get_output()
    
    lower_bound = 500
    upper_bound = 1000

    for i in tqdm(range(steps)):
        control_signal = 0.05
        if i >= 144:
            set_point = 16
            control_signal = 1
        if i >= 300:
            control_signal = 0.05
        # if i == 252:
        #     set_point = 16
        error = (outputs['temRoo.T'] - 273.15) - set_point
        total_error += (-error)
        cumulative_error += abs(error)
        delta_error = (-error) - last_error
        heat_P_power = outputs['heaPum.P']/5000
        
        # control_signal = kp * (-error) + ki * 300 * total_error + (kd/300) * delta_error
        heat_P_power += control_signal
        # control_signal = max(0, min(1, heat_P_power))
        
        
        control = {}
        control['u'] = [control_signal]
        outputs = env.step(control)
        out_list.append(outputs)
        
        last_error = -error
        
    out_df = pd.DataFrame(out_list)
    print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
    print(f"Mean temperature error: {cumulative_error/steps}")
    
    f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(15,20))

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
    
    ax3.plot(out_df['heaPum.COP'], 'b')
    ax3.set_ylabel('COP')
    ax3.set_xlabel('Steps')
    
    ax4.plot(out_df['temSup.T']-273.15, 'y')
    ax4.set_ylabel('Supply Temp')
    ax4.set_xlabel('Steps')
    
    ax5.plot(out_df["rad.Q_flow"], 'c')

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    # plt.savefig("./graphs/PID")
    plt.show()