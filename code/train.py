import gymnasium as gym
import registration
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import pandas as pd
import matplotlib.pyplot as plt
from utils import CustomCallback, algorithms

if __name__ == "__main__":
    e = [28]
    n = [36]
    num_cpu = 1
    alg = algorithms['PPO']
    
    for epochs, n_trainings in zip(e, n):
        out_list = []
        name = f"{alg['alg_name']}_{epochs}x{n_trainings}"
        
        vec_env = DummyVecEnv([lambda: gym.make("SwissHouseRSlaW2W-v0", action_type=alg['action_type']) for _ in range(num_cpu)])
        model = alg['alg']("MlpPolicy", vec_env, **alg['params'])
        model.policy = ActorCriticPolicy.load("./policy/PID_policy")
        # model = alg['alg'].load("trained_models/ddpg_pretrained", env=vec_env)
        for i in range(n_trainings):
            vec_env.env_method("set_random_start_day")
            print(f"Training {i+1}/{n_trainings}")
            model.learn(total_timesteps=288*epochs, callback=CustomCallback(out_list, vec_env), progress_bar=True)
        
        model.policy.save(f"policy/ppo_trained_policy_26x36")
        
        vec_env.close()
        out_df = pd.DataFrame(out_list)
        out_df.to_excel(f"datasets/ppo_trained_{epochs}x{n_trainings}.xlsx")
        
        f, (ax1, ax2) = plt.subplots(2, figsize=(12, 15))
        
        ax1.plot(out_df["temRoo.T"]-273.15, color='b', label='Room Temperature')
        ax1.axhline(y=16, color='r', linestyle='--')
        
        ax2.plot(out_df["heaPum.P"], color='b', label='Heat Pump Power')

        plt.legend()
        plt.savefig(f"graphs/ppo_trained_{epochs}x{n_trainings}.png")
        
plt.show()