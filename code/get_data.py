from matplotlib import pyplot as plt
import gymnasium as gym
import energym
import d3rlpy
import pandas as pd
import numpy as np
from tqdm import tqdm
import registration
from utils import unnormalize, weathers

if __name__ == "__main__":
    df_ddpg = pd.read_excel("./datasets/ddpg_only_online_10000x200k.xlsx")
    df_pid = pd.read_excel("./datasets/PID_test.xlsx")
    
    for i in range(28):
        print(f"Day {i+1}")
        for df in [df_ddpg, df_pid]:
            total_energy = df['heaPum.P'][i*288:(i+1)*288]
            total_energy = sum(total_energy)
            mean_power_consumption = total_energy/288
            print(mean_power_consumption)
            
            delta = abs(df['delta'][i*288:(i+1)*288])
            total_delta = sum(delta)
            mean_delta = total_delta/288
            print(f"{mean_delta}")