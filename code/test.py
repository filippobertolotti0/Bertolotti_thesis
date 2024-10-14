import gymnasium as gym
import registration
import d3rlpy
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import normalize, unnormalize, weathers, algorithms
import numpy as np

if __name__ == "__main__":
    out_list = []
    num_cpu = 1
    weather = weathers[4]

    env = gym.make("SwissHouseRSlaW2W-v0", weather=weather, eval_mode=False)
    model = d3rlpy.algos.DDPGConfig().create()
    model.build_with_env(env)
    model.load_model("./code/d3rlpy/trained_models/ddpg_after_bc")
    
    obs, _ = env.reset()
    steps = 4000
    cumulative_error = 0
    
    lower_bound = 500
    upper_bound = 1000
    env.set_set_point(289.15)
    
    for i in tqdm(range(steps)):
        if i == 1000:
            env.set_set_point(293.15)
            # lower_bound = 1250
            # upper_bound = 2000
        if i == 3000:
            env.set_set_point(289.15)
            # lower_bound = 2250
            # upper_bound = 3000
        # if i == 3000:
        #     env.set_set_point(295.15)
            # lower_bound = 3250
            # upper_bound = 4000
        # is_between = lower_bound < i < upper_bound
        action = model.predict(np.expand_dims(obs, axis=0))[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, rewards, terminated, truncated, info = env.step(action)
        cumulative_error += abs(obs[3])
        out_list.append({
                "heaPum.P": unnormalize(obs[0], 0, 1000),
                "temSup.T": unnormalize(obs[1], 273.15, 353.15),
                "TOut.T": unnormalize(obs[2], 253.15, 343.15),
                "temRoo.T": info['temRoo.T'],
            }
        )
        
    env.close()
    
    out_df = pd.DataFrame(out_list)
    print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
    print(f"Mean temperature error: {cumulative_error/steps}")
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
    
    ax1.plot(out_df["temRoo.T"]-273.15, color='b', label='Room Temperature')
    # ax1.plot(out_df["temSup.T"]-273.15, color='g', label='Supply Temperature')
    ax1.axhline(y=16, color='r', linestyle='--')
    # ax1.axhline(y=19, color='r', linestyle='--')
    ax1.axhline(y=20, color='r', linestyle='--')
    # ax1.axhline(y=22, color='r', linestyle='--')
    ax1.set_ylabel('Temp')
    ax1.set_xlabel('Steps')
    
    ax2.plot(out_df["heaPum.P"], color='b', label='Heat Pump Power')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Steps')
    
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    # plt.savefig(f"./graphs/{alg['alg_name']}_test_90x30")
    plt.show()

# if __name__ == "__main__":
#     out_list = []
#     num_cpu = 1
#     alg = algorithms['DDPG']
#     weather = weathers[4]

#     env = gym.make("SwissHouseRSlaW2W-v0", weather=weather, action_type=alg['action_type'], eval_mode=False)
#     # model = alg['alg'].load(f"trained_models/a2c_trained_336x1", env=env)
#     model = alg['alg']("MlpPolicy", env, **alg['params'])
#     model.policy = ActorCriticPolicy.load("./policy/PID_policy")
#     # model = DDPG.load(f"trained_models/{alg['alg_name']}_pretrained_test", env=env)
    
#     obs, _ = env.reset()
#     steps = 4000
#     cumulative_error = 0
    
#     lower_bound = 500
#     upper_bound = 1000
#     env.set_set_point(289.15)
    
#     for i in tqdm(range(steps)):
#         if i == 1000:
#             env.set_set_point(294.15)
#             # lower_bound = 1250
#             # upper_bound = 2000
#         elif i == 2000:
#             env.set_set_point(292.15)
#             # lower_bound = 2250
#             # upper_bound = 3000
#         if i == 3000:
#             env.set_set_point(295.15)
#             # lower_bound = 3250
#             # upper_bound = 4000
#         # is_between = lower_bound < i < upper_bound
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, terminated, truncated, info = env.step(action)
#         cumulative_error += abs(obs[3])
#         out_list.append({
#                 "heaPum.P": unnormalize(obs[0], 0, 1000),
#                 "temSup.T": unnormalize(obs[1], 273.15, 353.15),
#                 "TOut.T": unnormalize(obs[2], 253.15, 343.15),
#                 "temRoo.T": info['temRoo.T'],
#             }
#         )
        
#     env.close()
    
#     out_df = pd.DataFrame(out_list)
#     print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
#     print(f"Mean temperature error: {cumulative_error/2750}")
    
#     f, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
    
#     ax1.plot(out_df["temRoo.T"]-273.15, color='b', label='Room Temperature')
#     ax1.plot(out_df["temSup.T"]-273.15, color='g', label='Supply Temperature')
#     ax1.axhline(y=16, color='r', linestyle='--')
#     ax1.axhline(y=19, color='r', linestyle='--')
#     ax1.axhline(y=21, color='r', linestyle='--')
#     ax1.axhline(y=22, color='r', linestyle='--')
#     ax1.set_ylabel('Temp')
#     ax1.set_xlabel('Steps')
    
#     ax2.plot(out_df["heaPum.P"], color='b', label='Heat Pump Power')
#     ax2.set_ylabel('Energy')
#     ax2.set_xlabel('Steps')
    
#     plt.subplots_adjust(hspace=0.4)
#     plt.tight_layout()
#     # plt.savefig(f"./graphs/{alg['alg_name']}_test_90x30")
#     plt.show()