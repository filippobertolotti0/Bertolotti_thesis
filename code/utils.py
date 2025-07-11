import pandas as pd
from tqdm import tqdm

DAY = 288
WEEK = DAY*7
MONTH = WEEK*4
HALF_YEAR = MONTH*6
YEAR = MONTH*12

def normalize(x, min, max):
    return (x - min) / (max - min)

def unnormalize(x, min, max):
    return x * (max - min) + min

def get_dataset(path):
    df = pd.read_excel(path)
        
    obs = []
        
    for _, row in tqdm(df.iterrows()):
        row['heaPum.P'] = normalize(row['heaPum.P'], 0, 5000)
        row['temSup.T'] = normalize(row['temSup.T'], 273.15, 353.15)
        row['TOut.T'] = normalize(row['TOut.T'], 253.15, 343.15)
        row['delta'] = row['delta']
        
        obs.append([row['heaPum.P'], row['temSup.T'], row['TOut.T'], row['delta']])
    
    next_obs = obs[1:len(obs)]
    obs.pop()
        
    acts = df['acts'].values
    rewards = df['rewards'].values
    
    return obs, next_obs, acts, rewards

def save_training_params(model_name, training_params):
    with open(f"./trained_models/{model_name}/training_params.txt", "w") as file:
        for key, value in training_params.items():
            file.write(f"{key} = {value}\n")
    
weathers = ["CH_BS_Basel", "CH_ZH_Maur", "CH_TI_Bellinzona", "CH_GR_Davos", "CH_GE_Geneva", "CH_VD_Lausanne"]

DDPG_PARAMS_ONLINE = {
    'gamma': 0.95,
    'actor_learning_rate': 0.0007,
    'critic_learning_rate': 0.0002,
    'batch_size': 2048,
    'tau': 0.05,
    # 'update_interval': 8,
    # 'n_updates': 1
}

DDPG_PARAMS_OFFLINE = {
    'gamma': 0.88,
    'actor_learning_rate': 9.34293426e-05,
    'critic_learning_rate': 0.0008577842194694033,
    'batch_size': 256,
    'tau': 0.05,
    'n_steps': 400000,
    'n_steps_per_epoch': 50000,
    'episode_lenght': 288
}

TD3_PARAMS_ONLINE = {
    'gamma': 0.9,
    'actor_learning_rate': 0.0005965698722229415,
    'critic_learning_rate': 0.00013032913625000625,
    'batch_size': 2048,
    'tau': 0.05,
    'target_smoothing_sigma': 0.2,
    'target_smoothing_clip': 0.5,
    'update_actor_interval': 2,
    # 'update_interval': 9,
    # 'n_updates': 2
}

TD3_PARAMS_OFFLINE = {
    'actor_learning_rate': 0.0003899037287948072,
    'critic_learning_rate': 0.00023468554130439816,
    'batch_size': 128,
    'tau': 0.001,
    'gamma': 0.88,
    'target_smoothing_sigma': 0.5,
    'target_smoothing_clip': 0.3,
    'update_actor_interval': 4,
    'n_steps': 400000,
    'n_steps_per_epoch': 100000
}

SAC_PARAMS_ONLINE = {
    'actor_learning_rate': 0.000993354483407106,
    'batch_size': 2048,
    'critic_learning_rate': 2.6679098378465916e-05,
    'gamma': 0.88,
    'n_critics': 4,
    'tau': 0.02,
    'temp_learning_rate': 0.0002015096624240672,
    # 'update_interval': 8,
    # 'n_updates': 1,
}

SAC_PARAMS_OFFLINE = {
    'actor_learning_rate': 0.00013252793344665855,
    'critic_learning_rate': 1.190008670186549e-05,
    'temp_learning_rate': 1.1222270343816039e-05,
    'gamma': 0.88,
    'batch_size': 512,
    'tau': 0.05,
    'n_steps': 50000,
    'n_steps_per_epoch': 20000,
    'n_critics': 2
}

ONLINE_RESULTS = {
        ('1m', 'DDPG'): {'P': [1718.196, 1498.669, 1675.491, 1710.214, 1800.569],
                        'ΔT': [0.85, 1.371, 0.739, 0.739, 0.795]},
        ('1m', 'TD3'): {'P': [1712.904, 1640.283, 1742.459, 1709.847, 1732.403],
                        'ΔT': [0.577, 0.859, 0.553, 0.634, 0.581]},
        ('1m', 'SAC'): {'P': [2655.09, 2604.553, 2487.836, 1834.386, 2378.349],
                        'ΔT': [5.993, 6.337, 4.739, 2.686, 3.966]},
        
        ('2m', 'DDPG'): {'P': [1644.736, 1671.662, 1624.96, 1580.391, 1778.26],
                        'ΔT': [0.903, 0.754, 0.968, 1.092, 0.66]},
        ('2m', 'TD3'): {'P': [1771.922, 1428.868, 1755.936, 1666.422, 1677.204],
                        'ΔT': [0.651, 1.773, 0.942, 0.756, 0.809]},
        ('2m', 'SAC'): {'P': [1665.017, 1029.111, 1130.87, 1538.218, 1396.46],
                        'ΔT': [2.888, 5.02, 4.625, 1.92, 2.765]},

        ('3m', 'DDPG'): {'P': [1687.06, 1716.185, 1659.437, 1605.862, 1778.411],
                        'ΔT': [0.776, 1.129, 0.835, 1.0884, 0.505]},
        ('3m', 'TD3'): {'P': [1692.901, 1829.83, 1647.666, 1648.741, 1694.304],
                        'ΔT': [0.69, 0.739, 0.9, 0.85, 0.649]},
        ('3m', 'SAC'): {'P': [1383.381, 1600.818, 1316.564, 1698.028, 1672.086],
                        'ΔT': [2.263, 1.435, 2.808, 0.99, 1.134]},

        ('6m', 'DDPG'): {'P': [1730.024, 1651.727, 1734.284, 1727.193, 1678.207],
                        'ΔT': [0.698, 0.997, 0.57, 0.527, 0.752]},
        ('6m', 'TD3'): {'P': [1723.398, 1802.745, 1699.142, 1673.946, 1755.929],
                        'ΔT': [0.617, 0.892, 0.633, 0.773, 0.551]},
        ('6m', 'SAC'): {'P': [1638.281, 1135.701, 1697.253, 1699.855, 1611.562],
                        'ΔT': [0.934, 3.925, 0.754, 0.956, 1.207]},

        ('12m', 'DDPG'): {'P': [1738.089, 1743.651, 1708.598, 1744.324, 1754.079],
                        'ΔT': [0.734, 0.549, 0.577, 0.462, 0.493]},
        ('12m', 'TD3'): {'P': [1771.27, 1759.874, 1719.656, 1689.597, 1751.581],
                        'ΔT': [0.486, 0.415, 0.578, 0.672, 0.457]},
        ('12m', 'SAC'): {'P': [1715.575, 1681.326, 1725.018, 1753.409, 1754.249],
                        'ΔT': [0.684, 0.814, 0.681, 0.564, 0.528]},
    }

OFFLINE_RESULTS = {
    ('1m', 'DDPG'): {'P': [1594.549, 1627.751, 1742.981, 1651.668, 1759.539],
                    'ΔT': [1.915, 2.377, 0.828, 1.221, 0.753]},
    ('1m', 'TD3'): {'P': [1740.193, 1771.973, 1805.757, 1842.504, 1792.013],
                    'ΔT': [0.766, 0.944, 0.838, 0.879, 0.805]},
    ('1m', 'SAC'): {'P': [1848.446, 1896.457, 1847.726, 1879.812, 1948.658],
                    'ΔT': [1.046, 1.197, 0.942, 1.111, 1.479]},

    ('2m', 'DDPG'): {'P': [1521.288, 1540.131, 1217.483, 1617.461, 1774.052],
                    'ΔT': [1.469, 2.281, 4.744, 2.425, 0.659]},
    ('2m', 'TD3'): {'P': [1748.97, 1719.884, 1147.958, 1356.545, 1827.135],
                    'ΔT': [0.778, 0.849, 5.993, 4.964, 0.855]},
    ('2m', 'SAC'): {'P': [1864.591, 1535.422, 1890.595, 1857.045, 1900.07],
                    'ΔT': [1.107, 0.944, 1.29, 1.003, 1.113]},

    ('3m', 'DDPG'): {'P': [1575.389, 1538.173, 1758.965, 3793.475, 1651.189],
                    'ΔT': [1.135, 1.591, 1.284, 8.546, 1.492]},
    ('3m', 'TD3'): {'P': [1715.141, 1801.406, 1514.295, 830.75, 1765.904],
                    'ΔT': [0.867, 0.884, 4.499, 7.909, 0.923]},
    ('3m', 'SAC'): {'P': [2027.732, 1850.83, 1927.773, 1844.715, 1924.289],
                    'ΔT': [1.745, 0.923, 1.357, 0.907, 1.299]},

    ('6m', 'DDPG'): {'P': [1614.292, 1450.689, 1573.4, 1711.253, 2377.018],
                    'ΔT': [0.939, 2.392, 1.627, 2.186, 3.473]},
    ('6m', 'TD3'): {'P': [1805.453, 1856.806, 1979.03, 1409.766, 1569.97],
                    'ΔT': [0.929, 1.275, 1.334, 2.861, 1.985]},
    ('6m', 'SAC'): {'P': [1893.187, 1988.086, 1869.0, 1899.351, 1839.172],
                    'ΔT': [1.235, 1.524, 1.117, 1.142, 0.834]},

    ('12m', 'DDPG'): {'P': [1430.324, 1557.849, 1623.359, 1766.851, 4857.686],
                     'ΔT': [2.015, 2.655, 1.625, 2.235, 13.33]},
    ('12m', 'TD3'): {'P': [1818.556, 1967.568, 1681.269, 1848.059, 1841.354],
                     'ΔT': [0.877, 1.667, 1.48, 1.244, 1.519]},
    ('12m', 'SAC'): {'P': [2026.648, 1926.119, 1961.97, 1918.826, 1910.436],
                     'ΔT': [1.58, 1.358, 1.455, 1.202, 1.26]},
}

OFF_ON_RESULTS = {
    ('1w', 'DDPG'): {'P': [1667.452, 1703.326, 1744.228, 1680.217, 1689.945], 'ΔT': [0.762, 0.644, 0.898, 0.994, 0.76]},
    ('1w', 'TD3'): {'P': [1675.93, 1692.876, 1710.097, 1748.968, 1704.242], 'ΔT': [0.708, 0.631, 0.828, 0.651, 0.645]},
    ('1w', 'SAC'): {'P': [1579.921, 968.949, 2297.021, 1799.122, 1703.331], 'ΔT': [1.847, 6.953, 3.618, 1.459, 1.338]},

    ('2w', 'DDPG'): {'P': [1685.051, 1683.884, 1751.954, 1787.483, 1745.448], 'ΔT': [0.722, 0.708, 0.695, 0.609, 0.607]},
    ('2w', 'TD3'): {'P': [1688.375, 1660.392, 1679.755, 1747.365, 1739.368], 'ΔT': [0.679, 0.765, 0.82, 0.53, 0.546]},
    ('2w', 'SAC'): {'P': [1658.231, 1634.794, 1797.235, 1783.579, 1739.765], 'ΔT': [0.91, 0.964, 0.678, 0.739, 0.601]},
    
    ('1m', 'DDPG'): {'P': [1743.612, 1736.222, 1773.566, 1740.063, 1739.144], 'ΔT': [0.582, 0.745, 0.652, 0.639, 0.579]},
    ('1m', 'TD3'): {'P': [1791.522, 1720.988, 1775.54, 1716.912, 1748.098], 'ΔT': [0.712, 0.719, 0.445, 0.658, 0.593]},
    ('1m', 'SAC'): {'P': [1693.678, 1678.417, 1797.728, 1773.713, 1792.838], 'ΔT': [0.714, 0.78, 0.555, 0.61, 0.563]},

    ('2m', 'DDPG'): {'P': [1711.295, 1748.508, 1701.353, 1747.673, 1707.115], 'ΔT': [0.699, 0.714, 0.612, 0.568, 0.672]},
    ('2m', 'TD3'): {'P': [1659.705, 1654.881, 1740.145, 1701.639, 1646.417], 'ΔT': [0.857, 0.846, 0.695, 0.794, 0.924]},
    ('2m', 'SAC'): {'P': [1762.595, 1658.818, 1608.74, 1699.969, 1681.916], 'ΔT': [0.465, 0.905, 1.104, 0.783, 0.726]},

    ('3m', 'DDPG'): {'P': [1765.077, 1757.435, 1741.679, 1736.453, 1749.333], 'ΔT': [0.539, 0.509, 0.628, 0.527, 0.502]},
    ('3m', 'TD3'): {'P': [1729.171, 1711.94, 1724.582, 1718.68, 1673.047], 'ΔT': [0.579, 0.809, 0.531, 0.699, 0.808]},
    ('3m', 'SAC'): {'P': [1813.122, 1746.714, 1638.043, 1721.296, 1780.368], 'ΔT': [0.691, 0.758, 0.958, 0.721, 0.611]},
}

PID_RESULTS = {
    'P': 1766.284,
    'ΔT': 0.328,
}

dati_complessivi_rl = {
    'SAC': 40.01,
    'DDPG': 14.42,
    'TD3': 11.09
}

# Dati del PID corrispondenti a ciascun esperimento. 
# Nota: il PID ha performance leggermente diverse nei due set di esperimenti.
# Potresti plottare una media o la linea di riferimento più rilevante.
dati_complessivi_pid = {
    'PID (vs SAC/DDPG)': 14.60,
    'PID (vs TD3)': 12.38
}


# ==============================================================================
# DATI PER LA FUNZIONE 2: GRAFICI MENSILI COMPARATIVI
# ==============================================================================
# Ogni dizionario seguente contiene tutto il necessario per creare un grafico
# che confronta un modello RL con il PID, mese per mese.
# - 'mesi': etichette per l'asse X.
# - 'rl_percentuali': barre dei risultati per l'algoritmo RL.
# - 'pid_percentuali': barre dei risultati per il controller PID.

# --- Dati per il modello: sac_6m_22 ---
dati_mensili_sac = {
    "nome_modello": "SAC",
    "mesi": ["Mese 1", "Mese 2", "Mese 3", "Mese 4", "Mese 5", "Mese 6"],
    "rl_percentuali": [87.64, 59.19, 28.91, 24.07, 21.06, 19.22],
    "pid_percentuali": [24.28, 13.99, 12.76, 12.39, 12.10, 12.10],
    "color": "#55a868",
}

# --- Dati per il modello: ddpg_6m_22 ---
dati_mensili_ddpg = {
    "nome_modello": "DDPG",
    "mesi": ["Mese 1", "Mese 2", "Mese 3", "Mese 4", "Mese 5", "Mese 6"],
    "rl_percentuali": [65.08, 2.37, 8.28, 5.44, 2.59, 2.78],
    "pid_percentuali": [24.28, 13.99, 12.76, 12.39, 12.10, 12.10],
    "color":"#4c72b0"
}

# --- Dati per il modello: td3_6m_20 ---
dati_mensili_td3 = {
    "nome_modello": "TD3",
    "mesi": ["Mese 1", "Mese 2", "Mese 3", "Mese 4", "Mese 5", "Mese 6"],
    "rl_percentuali": [47.19, 4.02, 3.89, 5.03, 3.11, 3.27],
    "pid_percentuali": [13.16, 12.44, 12.14, 12.54, 11.98, 12.04],
    "color":"#dd8452"
}



# ==============================================================================
# DATI PER GLI ESPERIMENTI "OFFLINE-ONLINE"
# ==============================================================================

# ------------------------------------------------------------------------------
# DATI PER LA FUNZIONE 1: GRAFICO COMPLESSIVO (Offline-Online)
# ------------------------------------------------------------------------------
# Performance totale dei modelli addestrati offline e poi riaddestrati online.

dati_complessivi_rl_off_on = {
    'DDPG': 11.63,
    'TD3': 8.52,
    'SAC': 33.93
}

# Dati del PID corrispondenti a ciascun esperimento "offline-online".
# Nota: anche qui ci sono due baseline PID diverse.
dati_complessivi_pid_off_on = {
    'PID (vs DDPG/TD3)': 12.34,
    'PID (vs SAC)': 13.16
}


# ------------------------------------------------------------------------------
# DATI PER LA FUNZIONE 2: GRAFICI MENSILI COMPARATIVI (Offline-Online)
# ------------------------------------------------------------------------------
# Dati mensili per ciascun esperimento "offline-online".
# Nota: questi esperimenti durano 4 mesi.

# --- Dati per il modello: ddpg_off_on_1m_feb ---
dati_settimanali_ddpg_off_on = {
    "nome_modello": "DDPG",
    "mesi": ["Week 1", "Week 2", "Week 3", "Week 4"],
    "rl_percentuali": [27.78, 6.30, 10.76, 1.69],
    "pid_percentuali": [13.00, 12.05, 12.00, 12.31],
    "color":"#4c72b0"
}

# --- Dati per il modello: td3_off_on_1m_feb ---
dati_settimanali_td3_off_on = {
    "nome_modello": "TD3",
    "mesi": ["Week 1", "Week 2", "Week 3", "Week 4"],
    "rl_percentuali": [28.62, 2.23, 1.04, 2.18],
    "pid_percentuali": [13.00, 12.05, 12.00, 12.31],
    "color":"#dd8452"
}

# --- Dati per il modello: sac_off_on_1m_oct ---
dati_settimanali_sac_off_on = {
    "nome_modello": "SAC",
    "mesi": ["Week 1", "Week 2", "Week 3", "Week 4"],
    "rl_percentuali": [64.48, 40.18, 16.37, 14.69],
    "pid_percentuali": [15.08, 12.50, 12.40, 12.66],
    "color": "#55a868",
}

off_on_data = [
    # --- Group 1: Offline Pre-training + Online Fine-tuning ---
    {
        "model_name": "DDPG (Offline+Online)",
        "type": "Offline+Online",
        "total_energy_saving_W": 565479.82,
        "total_bad_steps": {
            "rl": {"count": 983, "percentage": 12.19},
            "pid": {"count": 988, "percentage": 12.25}
        },
        "time_series_data": [
            # NOTE: Corrected all to 'Week' as requested
            {"period_label": "Week 1", "energy_saving_W": 180951.52, "quadrants": {"good_control_saving": 17.81, "good_control_waste": 10.81, "bad_control_waste": 27.98, "bad_control_saving": 43.40}},
            {"period_label": "Week 2", "energy_saving_W": 190763.65, "quadrants": {"good_control_saving": 37.50, "good_control_waste": 16.07, "bad_control_waste": 7.29,  "bad_control_saving": 39.14}},
            {"period_label": "Week 3", "energy_saving_W": 128893.37, "quadrants": {"good_control_saving": 45.73, "good_control_waste": 21.83, "bad_control_waste": 11.26, "bad_control_saving": 21.18}},
            {"period_label": "Week 4", "energy_saving_W": 64871.28,  "quadrants": {"good_control_saving": 52.80, "good_control_waste": 34.39, "bad_control_waste": 2.23,  "bad_control_saving": 10.57}},
        ]
    },
    {
        "model_name": "TD3 (Offline+Online)",
        "type": "Offline+Online",
        "total_energy_saving_W": 287246.71,
        "total_bad_steps": {
            "rl": {"count": 772, "percentage": 9.57},
            "pid": {"count": 988, "percentage": 12.25}
        },
        "time_series_data": [
            {"period_label": "Week 1", "energy_saving_W": 31611.83,  "quadrants": {"good_control_saving": 18.60, "good_control_waste": 18.55, "bad_control_waste": 29.02, "bad_control_saving": 33.83}},
            {"period_label": "Week 2", "energy_saving_W": 133269.92, "quadrants": {"good_control_saving": 39.09, "good_control_waste": 28.37, "bad_control_waste": 3.57,  "bad_control_saving": 28.97}},
            {"period_label": "Week 3", "energy_saving_W": 68586.52,  "quadrants": {"good_control_saving": 51.59, "good_control_waste": 29.17, "bad_control_waste": 2.88,  "bad_control_saving": 16.37}},
            {"period_label": "Week 4", "energy_saving_W": 53778.44,  "quadrants": {"good_control_saving": 56.97, "good_control_waste": 30.37, "bad_control_waste": 2.83,  "bad_control_saving": 9.83}},
        ]
    },
    {
        "model_name": "SAC (Offline+Online)",
        "type": "Offline+Online",
        "total_energy_saving_W": -938236.67,
        "total_bad_steps": {
            "rl": {"count": 1760, "percentage": 21.83},
            "pid": {"count": 988, "percentage": 12.25}
        },
        "time_series_data": [
            {"period_label": "Week 1", "energy_saving_W": -760050.59, "quadrants": {"good_control_saving": 9.33,  "good_control_waste": 8.58,  "bad_control_waste": 50.00, "bad_control_saving": 32.09}},
            {"period_label": "Week 2", "energy_saving_W": 5885.95,    "quadrants": {"good_control_saving": 35.47, "good_control_waste": 31.89, "bad_control_waste": 15.53, "bad_control_saving": 17.11}},
            {"period_label": "Week 3", "energy_saving_W": -81210.65,  "quadrants": {"good_control_saving": 39.88, "good_control_waste": 38.84, "bad_control_waste": 9.33,  "bad_control_saving": 11.95}},
            {"period_label": "Week 4", "energy_saving_W": -102861.37, "quadrants": {"good_control_saving": 34.49, "good_control_waste": 36.13, "bad_control_waste": 12.46, "bad_control_saving": 16.92}},
        ]
    },
    # {
    #     "model_name": "PID",
    #     "type": "",
    #     "total_energy_saving_W": 0,
    #     "total_bad_steps": {},
    #     "time_series_data": [
    #         {"period_label": "Week 1", "energy_saving_W": 0, "quadrants": {"good_control_saving": 87.15, "good_control_waste": 0, "bad_control_waste": 12.85, "bad_control_saving": 0}},
    #         {"period_label": "Week 2", "energy_saving_W": 0,   "quadrants": {"good_control_saving": 87.95, "good_control_waste": 0, "bad_control_waste": 12.05,  "bad_control_saving": 0}},
    #         {"period_label": "Week 3", "energy_saving_W": 0,   "quadrants": {"good_control_saving": 88.00, "good_control_waste": 0, "bad_control_waste": 12.00,  "bad_control_saving": 0}},
    #         {"period_label": "Week 4", "energy_saving_W": 0,   "quadrants": {"good_control_saving": 87.89, "good_control_waste": 0, "bad_control_waste": 12.11,  "bad_control_saving": 0}}
    #     ]
    # }
]

on_data = [
    {
        "model_name": "DDPG (Online Only)",
        "type": "Online Only",
        "total_energy_saving_W": -5968999.49,
        "total_bad_steps": {
            "rl": {"count": 7243, "percentage": 14.97},
            "pid": {"count": 7033, "percentage": 14.54}
        },
        "time_series_data": [
            {"period_label": "Month 1", "energy_saving_W": -8620984.25, "quadrants": {"good_control_saving": 12.13, "good_control_waste": 13.17, "bad_control_waste": 64.71, "bad_control_saving": 10.00}},
            {"period_label": "Month 2", "energy_saving_W": 198019.70,   "quadrants": {"good_control_saving": 52.43, "good_control_waste": 32.08, "bad_control_waste": 3.11,  "bad_control_saving": 12.38}},
            {"period_label": "Month 3", "energy_saving_W": 1030174.14,  "quadrants": {"good_control_saving": 27.37, "good_control_waste": 5.03,  "bad_control_waste": 9.06,  "bad_control_saving": 58.53}},
            {"period_label": "Month 4", "energy_saving_W": 593599.88,   "quadrants": {"good_control_saving": 50.32, "good_control_waste": 16.58, "bad_control_waste": 6.31,  "bad_control_saving": 26.79}},
            {"period_label": "Month 5", "energy_saving_W": 402042.22,   "quadrants": {"good_control_saving": 52.54, "good_control_waste": 34.72, "bad_control_waste": 3.19,  "bad_control_saving": 9.55}},
            {"period_label": "Month 6", "energy_saving_W": 428148.82,   "quadrants": {"good_control_saving": 62.98, "good_control_waste": 23.84, "bad_control_waste": 3.44,  "bad_control_saving": 9.75}},
        ]
    },
    {
        "model_name": "SAC (Online Only)",
        "type": "Online Only",
        "total_energy_saving_W": -15855019.77,
        "total_bad_steps": {
            "rl": {"count": 19389, "percentage": 40.07},
            "pid": {"count": 7033, "percentage": 14.54}
        },
        "time_series_data": [
            {"period_label": "Month 1", "energy_saving_W": -15637184.46, "quadrants": {"good_control_saving": 0.00,  "good_control_waste": 0.00,  "bad_control_waste": 87.71, "bad_control_saving": 12.29}},
            {"period_label": "Month 2", "energy_saving_W": -5784009.24,  "quadrants": {"good_control_saving": 6.55,  "good_control_waste": 5.12,  "bad_control_waste": 59.25, "bad_control_saving": 29.08}},
            {"period_label": "Month 3", "energy_saving_W": 3354973.31,   "quadrants": {"good_control_saving": 10.04, "good_control_waste": 8.26,  "bad_control_waste": 28.87, "bad_control_saving": 52.83}},
            {"period_label": "Month 4", "energy_saving_W": 934012.32,    "quadrants": {"good_control_saving": 25.05, "good_control_waste": 20.29, "bad_control_waste": 24.03, "bad_control_saving": 30.63}},
            {"period_label": "Month 5", "energy_saving_W": 710634.25,    "quadrants": {"good_control_saving": 29.38, "good_control_waste": 23.40, "bad_control_waste": 21.22, "bad_control_saving": 26.00}},
            {"period_label": "Month 6", "energy_saving_W": 566554.04,    "quadrants": {"good_control_saving": 32.51, "good_control_waste": 25.95, "bad_control_waste": 19.36, "bad_control_saving": 22.19}},
        ]
    },
    {
        "model_name": "TD3 (Online Only)",
        "type": "Online Only",
        "total_energy_saving_W": -5604769.57,
        "total_bad_steps": {
            "rl": {"count": 5467, "percentage": 11.30},
            "pid": {"count": 5963, "percentage": 12.32}
        },
        "time_series_data": [
            {"period_label": "Month 1", "energy_saving_W": -6829160.77, "quadrants": {"good_control_saving": 17.92, "good_control_waste": 22.33, "bad_control_waste": 46.66, "bad_control_saving": 13.08}},
            {"period_label": "Month 2", "energy_saving_W": 124587.65,   "quadrants": {"good_control_saving": 54.49, "good_control_waste": 29.41, "bad_control_waste": 4.38,  "bad_control_saving": 11.72}},
            {"period_label": "Month 3", "energy_saving_W": 276301.09,   "quadrants": {"good_control_saving": 48.33, "good_control_waste": 35.76, "bad_control_waste": 4.06,  "bad_control_saving": 11.86}},
            {"period_label": "Month 4", "energy_saving_W": 735151.34,   "quadrants": {"good_control_saving": 45.46, "good_control_waste": 35.66, "bad_control_waste": 5.46,  "bad_control_saving": 13.42}},
            {"period_label": "Month 5", "energy_saving_W": 71114.75,    "quadrants": {"good_control_saving": 49.64, "good_control_waste": 35.33, "bad_control_waste": 3.48,  "bad_control_saving": 11.55}},
            {"period_label": "Month 6", "energy_saving_W": 17236.37,    "quadrants": {"good_control_saving": 45.80, "good_control_waste": 36.90, "bad_control_waste": 3.76,  "bad_control_saving": 13.54}},
        ]
    },
    # {
    #     "model_name": "PID",
    #     "type": "",
    #     "total_energy_saving_W": 0,
    #     "total_bad_steps": {},
    #     "time_series_data": [
    #         {"period_label": "Month 1", "energy_saving_W": 0, "quadrants": {"good_control_saving": 81.33, "good_control_waste": 0, "bad_control_waste": 18.67, "bad_control_saving": 0}},
    #         {"period_label": "Month 2", "energy_saving_W": 0,   "quadrants": {"good_control_saving": 86.83, "good_control_waste": 0, "bad_control_waste": 13.17,  "bad_control_saving": 0}},
    #         {"period_label": "Month 3", "energy_saving_W": 0,   "quadrants": {"good_control_saving": 87.58, "good_control_waste": 0, "bad_control_waste": 12.42,  "bad_control_saving": 0}},
    #         {"period_label": "Month 4", "energy_saving_W": 0,   "quadrants": {"good_control_saving": 87.60, "good_control_waste": 0, "bad_control_waste": 12.40,  "bad_control_saving": 0}},
    #         {"period_label": "Month 5", "energy_saving_W": 0,    "quadrants": {"good_control_saving": 88.01, "good_control_waste": 0, "bad_control_waste": 11.99,  "bad_control_saving": 0}},
    #         {"period_label": "Month 6", "energy_saving_W": 0,    "quadrants": {"good_control_saving": 88.05, "good_control_waste": 0, "bad_control_waste": 11.95,  "bad_control_saving": 0}},
    #     ]
    # }
]