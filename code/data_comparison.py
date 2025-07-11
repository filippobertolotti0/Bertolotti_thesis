from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import gymnasium as gym
import energym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import os
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from tqdm import tqdm
import registration
from utils import unnormalize, on_data, off_on_data, DAY, WEEK, MONTH, HALF_YEAR, YEAR, TD3_PARAMS_ONLINE, SAC_PARAMS_ONLINE, OFFLINE_RESULTS, ONLINE_RESULTS, OFF_ON_RESULTS, dati_complessivi_pid, dati_complessivi_rl, dati_mensili_ddpg, dati_mensili_sac, dati_mensili_td3, dati_complessivi_pid_off_on, dati_complessivi_rl_off_on, dati_settimanali_ddpg_off_on, dati_settimanali_sac_off_on, dati_settimanali_td3_off_on
from PID.PID import PID
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

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

def get_tick_positions(week_number: int):
    if week_number == 1:
        return range(0, 8064+1, 1152)
    elif week_number == 2:
        return range(8064, 16128+1, 1152)
    elif week_number == 3:
        return range(16128, 24192+1, 1152)
    elif week_number == 4:
        return range(24192, 32256+1, 1152)
    elif week_number == 5:
        return range(32256, 40320+1, 1152)
    elif week_number == 6:
        return range(40320, 48384+1, 1152)
    else:
        print("Invalid week number")
        
def get_week_data(df: pd.DataFrame, week_number: int):
    if week_number == 1:
        return df[:8064]
    elif week_number == 2:
        return df[8064:16128]
    elif week_number == 3:
        return df[16128:24192]
    elif week_number == 4:
        return df[24192:32256]
    elif week_number == 5:
        return df[32256:40320]
    elif week_number == 6:
        return df[40320:48384]
    else:
        print("Invalid week number")
    
def reward_convergence(path, episode_length, total_steps):
    plt.figure(figsize=(12, 6)) 
    for p in path:
        df = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["reward"]
        rewards = []
        step_number = []
        # total_step = len(df)
        total_step = total_steps
        total_episodes = total_step // episode_length
        
        for i in range(total_episodes):
            episode_reward = df[i*episode_length:(i+1)*episode_length]
            # mean_reward = sum(episode_reward)/episode_length
            cumulative_reward = sum(episode_reward)
            rewards.append(cumulative_reward)
            step_number.append(i+1)
        plt.plot(step_number, rewards, label=p)
        
    # for i in range(HALF_YEAR//episode_length, total_episodes, HALF_YEAR//episode_length):
    #     plt.axvline(x=i, color='g', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig("./reward_convergence_1y")
    plt.show()
    
def reward_convergence_splitted(path, episode_length):
    plt.figure(figsize=(12, 6)) 
    for p in path:
        temp_penalty = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["temp_penalty"]
        energy_penalty = pd.read_excel(f"./trained_models/{p}/{p}.xlsx")["energy_penalty"]
        temp_rewards = []
        energy_rewards = []
        step_number = []
        total_step = len(temp_penalty)
        total_episodes = total_step // episode_length
        
        for i in range(total_episodes):
            episode_temp_penalty = temp_penalty[i*episode_length:(i+1)*episode_length]
            episode_energy_penalty = energy_penalty[i*episode_length:(i+1)*episode_length]
            # mean_reward = sum(episode_reward)/episode_length
            temp_cumulative_reward = sum(episode_temp_penalty)
            energy_cumulative_reward = sum(episode_energy_penalty)
            temp_rewards.append(temp_cumulative_reward)
            energy_rewards.append(energy_cumulative_reward)
            step_number.append(i+1)
        plt.plot(step_number, temp_rewards, label=f"{p}: Temperature penalty")
        plt.plot(step_number, energy_rewards, label=f"{p}: Energy penalty")
        
    for i in range(HALF_YEAR//episode_length, total_episodes, HALF_YEAR//episode_length):
        plt.axvline(x=i, color='g', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.tight_layout()
    plt.savefig("./graphs/reward_convergence/reward_convergence_splitted")
    plt.legend()
    plt.show()
    
def moving_average(data, window_size):
    """Calcola la media mobile"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_avg_daily_rewards():
    algo = "td3"
    dataset_files = []

    labels = [
        f"{algo.upper()} 1 week",
        f"{algo.upper()} 2 weeks",
        f"{algo.upper()} 1 month",
        f"{algo.upper()} 2 months",
        f"{algo.upper()} 3 months"
    ]

    max_elements = 28 * 6

    # --- IMPOSTAZIONI DIMENSIONE FONT ---
    axis_label_fontsize = 13
    legend_fontsize = 13
    tick_labelsize = 14
    # --- FINE IMPOSTAZIONI ---
    
    plt.figure(figsize=(12, 6)) 

    for file, label in zip(dataset_files, labels):
        try:
            df = pd.read_excel(file)
            rewards = df["reward"].tolist()[:max_elements]

            if len(rewards) >= 1:
                smoothed = moving_average(rewards, 1)
            else:
                smoothed = np.array(rewards) # Assicurati che sia un array per la coerenza

            # Gli episodi dovrebbero corrispondere alla lunghezza dei dati smussati
            episodes = list(range(1, len(smoothed) + 1))
            plt.plot(episodes, smoothed, label=label)
        except FileNotFoundError:
            print(f"Attenzione: il file {file} non è stato trovato.")
            continue # Salta al prossimo file
        except Exception as e:
            print(f"Errore durante l'elaborazione del file {file}: {e}")
            continue


    plt.xlabel("Episode (Day)", fontsize=axis_label_fontsize)
    plt.ylabel("Average cumulative reward (moving average)", fontsize=axis_label_fontsize)
    plt.legend(fontsize=legend_fontsize)

    # Modifica la dimensione dei tick per entrambi gli assi
    plt.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    # --- Blocco per creare la cartella graphs se non esiste ---
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    # --- Fine blocco ---
    plt.savefig()
    plt.show()

def avg_reward_convergence(base_path, episode_length, total_steps, start_index=20, end_index=24, ma_window=10):
    algorithms = ["ddpg", "td3", "sac"]
    plt.figure(figsize=(12, 6)) 

    for algo in algorithms:
        all_rewards = []

        for i in range(start_index, end_index + 1):
            path = f"{base_path}/{algo}_6m_{i}"
            file_name = f"{algo}_6m_{i}.xlsx"
            df = pd.read_excel(f"{path}/{file_name}")["reward"]

            rewards = []
            total_episodes = total_steps // episode_length

            for ep in range(total_episodes):
                episode_reward = df[ep*episode_length:(ep+1)*episode_length]
                cumulative_reward = sum(episode_reward)
                rewards.append(cumulative_reward)

            all_rewards.append(rewards)

        mean_rewards = np.mean(all_rewards, axis=0)

        if len(mean_rewards) >= ma_window:
            smoothed = moving_average(mean_rewards, ma_window)
            episodes = list(range(1, len(smoothed) + 1))
        else:
            smoothed = mean_rewards
            episodes = list(range(1, len(smoothed) + 1))

        plt.plot(episodes, smoothed, label=algo.upper())

    plt.xlabel("Episode")
    plt.ylabel("Average cumulative reward (moving average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig()
    plt.show()
    #td3 -150
    #ddpg -300
    
def export_daily_avg_rewards(episode_length=288):
    for algo in ["ddpg", "td3", "sac"]:
        model_paths = []
        grouped_rewards = {}
        total_steps = DAY*7
        output_file = f"avg_rewards/{algo}_off_on_avg_rewards_1w.xlsx"

        for path in model_paths:
            file_name = [f for f in os.listdir(path) if f.endswith(".xlsx")][0]
            algo = file_name.split("_")[0]  # estrae "ddpg" da "ddpg_6m_20.xlsx"
            df = pd.read_excel(os.path.join(path, file_name))["reward"]

            total_episodes = total_steps // episode_length
            rewards = []

            for ep in range(total_episodes):
                episode_reward = df[ep * episode_length:(ep + 1) * episode_length]
                cumulative_reward = sum(episode_reward)
                rewards.append(cumulative_reward)

            if algo not in grouped_rewards:
                grouped_rewards[algo] = []
            grouped_rewards[algo].append(rewards)

        # Calcola la media per episodio (giornaliera) per ogni algoritmo
        avg_rewards_per_algo = {
            algo: np.mean(runs, axis=0)
            for algo, runs in grouped_rewards.items()
        }

        # Crea un DataFrame con le medie giornaliere
        df_out = pd.DataFrame(avg_rewards_per_algo)
        df_out.index.name = "Day"

        # Salva su file Excel
        df_out.to_excel(output_file)

        print(f"File salvato: {output_file}")

def training_analysis(window=7, offset=0):
    # Caricamento dei dati
    for model_name in []:
        df_rl = pd.read_excel()
        df_pid = pd.read_excel()

        # Validazione della lunghezza
        n = min(len(df_rl), len(df_pid))
        df_rl = df_rl.iloc[:n]
        df_pid = df_pid.iloc[:n]

        # Parametri
        chunk_size = 288  # un giorno = 288 step da 5 minuti
        daily_pid_power = []
        daily_rl_power = []
        daily_rl_discomfort = []

        # Calcolo potenza e discomfort giornalieri
        for i in range(0, n, chunk_size):
            chunk_rl_p = df_rl['heaPum.P'].iloc[i:i + chunk_size]
            chunk_rl_d = df_rl['delta'].iloc[i:i + chunk_size]
            chunk_pid_p = df_pid['heaPum.P'].iloc[i:i + chunk_size]

            if len(chunk_rl_p) < chunk_size:
                break  # blocco incompleto

            daily_rl_power.append(chunk_rl_p.mean())
            daily_rl_discomfort.append(chunk_rl_d.mean())
            daily_pid_power.append(chunk_pid_p.mean())

        # Trova il primo giorno con 5 consecutivi validi e conta i validi fino a quel punto
        counter = 0
        valid_days_until_target = 0

        for day in range(len(daily_rl_power)):
            power_rl = daily_rl_power[day]
            power_pid = daily_pid_power[day]
            discomfort = daily_rl_discomfort[day]

            is_valid = power_rl < power_pid and abs(discomfort) <= 0.7

            if is_valid:
                counter += 1
            else:
                counter = 0

            # Conta tutti i giorni validi fino al giorno in cui parte la sequenza di 5 validi
            if counter < 10 and is_valid:
                valid_days_until_target += 1

            if counter >= 10:
                print(f"{model_name}: Condizione raggiunta al giorno {day - 4} (dal giorno {day - 4} al {day})")
                print(f"Giorni validi totali fino a quel punto: {valid_days_until_target}")
                return day - 4, valid_days_until_target

        print(f"{model_name}: Nessuna sequenza di 5 giorni consecutivi trovata.")
        print(f"Giorni validi totali: {valid_days_until_target}")

def saving_momentum(model_name, window=7, offset=0):
    # Caricamento dei dati
    df_rl = pd.read_excel()
    df_pid = pd.read_excel()

    # Validazione della lunghezza
    n = min(len(df_rl), len(df_pid))
    df_rl = df_rl.iloc[:n]
    df_pid = df_pid.iloc[:n]

    # Parametri
    chunk_size = 1  # un giorno = 288 step da 5 minuti
    daily_percent_savings = []
    daily_avg_discomfort = []

    # Controllo colonne
    if 'heaPum.P' not in df_rl.columns or 'heaPum.P' not in df_pid.columns:
        raise ValueError("La colonna 'heaPum.P' deve essere presente in entrambi i DataFrame.")
    if 'delta' not in df_rl.columns:
        raise ValueError("La colonna 'delta' deve essere presente nel DataFrame RL.")

    # Calcolo risparmio e discomfort giornalieri
    for i in range(0, n, chunk_size):
        chunk_rl_p = df_rl['heaPum.P'].iloc[i:i + chunk_size]
        chunk_pid_p = df_pid['heaPum.P'].iloc[i:i + chunk_size]
        chunk_rl_d = df_rl['delta'].iloc[i:i + chunk_size]

        if len(chunk_rl_p) < chunk_size or len(chunk_pid_p) < chunk_size:
            break  # blocco incompleto

        mean_rl = chunk_rl_p.mean()
        mean_pid = chunk_pid_p.mean()
        mean_delta = chunk_rl_d.mean()

        if mean_pid > 0:
            percent_saving = ((mean_pid - mean_rl) / mean_pid) * 100
            daily_percent_savings.append(percent_saving)
            daily_avg_discomfort.append(mean_delta)

    # Rimozione dei primi 20 episodi
    savings = daily_percent_savings[offset:]
    discomfort = daily_avg_discomfort[offset:]

    # Media mobile
    savings_smoothed = pd.Series(savings).rolling(window, center=True).mean()
    discomfort_smoothed = pd.Series(discomfort).rolling(window, center=True).mean()

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(savings, color='green', label="Energy saving (%)")
    ax2.plot(discomfort, color='red', label="Delta Temperature (°C)")

    ax1.set_xlabel("Days")
    ax1.set_ylabel("Energy saving (%)", color='green')
    ax2.set_ylabel("Delta temperature (°C)", color='red')

    ax1.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.grid(True)
    # plt.title(f"Andamento Risparmio vs Discomfort - {model_name}")
    fig.tight_layout()
    plt.show()
    plt.savefig()
    
def analizza_performance_agente(model_name: str, dataset_name: str) -> dict:
    """
    Esegue un'analisi comparativa completa tra un agente RL e un PID, calcolando
    metriche di performance totali e con distribuzione mensile.

    L'analisi si basa su 4 quadranti che combinano il controllo della temperatura
    e l'efficienza energetica.

    Args:
        model_name (str): Il nome del modello RL da analizzare.
        dataset_name (str): Il nome del dataset di confronto (PID).

    Returns:
        dict: Un dizionario dettagliato con i risultati dell'analisi.
    """
    # --- 1. Caricamento e Validazione Dati ---
    try:
        df_rl = pd.read_excel()
        df_pid = pd.read_excel()
    except FileNotFoundError as e:
        print(f"ERRORE: File non trovato. Controlla i percorsi. Dettagli: {e}")
        return {}

    n = min(len(df_rl), len(df_pid))
    if n == 0:
        print("ERRORE: Uno dei DataFrame è vuoto.")
        return {}
        
    df_rl = df_rl.iloc[:n].copy()
    df_pid = df_pid.iloc[:n].copy()

    required_cols = ['heaPum.P', 'delta']
    if not all(col in df_rl.columns for col in required_cols) or \
       not all(col in df_pid.columns for col in required_cols):
        raise ValueError(f"Le colonne {required_cols} devono essere presenti in entrambi i DataFrame.")
    
    Prl = df_rl['heaPum.P']
    Ppid = df_pid['heaPum.P']
    delta_T_rl = df_rl['delta']
    delta_T_pid = df_pid['delta']

    # --- 2. Definizione dei Quadranti di Performance ---
    
    # Quadrante 1: Buon controllo, Risparmio (Caso Ideale)
    mask_buon_controllo_risparmio = (abs(delta_T_rl) <= 0.5) & (Prl <= Ppid)
    
    # Quadrante 2: Buon controllo, Spreco
    mask_buon_controllo_spreco = (abs(delta_T_rl) <= 0.5) & (Prl > Ppid)
    
    # Quadrante 3: Cattivo controllo, Spreco (Caso Peggiore)
    mask_cattivo_controllo_spreco = (abs(delta_T_rl) > 0.5) & (Prl > Ppid)
    
    # Quadrante 4: Cattivo controllo, Risparmio
    mask_cattivo_controllo_risparmio = (abs(delta_T_rl) > 0.5) & (Prl <= Ppid)
    
    # Metrica per il PID (per confronto)
    pessimi_step_mask_pid = abs(delta_T_pid) > 0.5

    # Calcolo totali
    risparmio_energetico_totale = (Ppid - Prl).sum()
    numero_pessimi_step_pid = int(pessimi_step_mask_pid.sum())

    # --- 3. Calcolo Distribuzioni Mensili ---
    STEPS_PER_MESE = 288 * 7
    risultati_mensili = {}
    
    all_masks = {
        'buon_controllo_risparmio': mask_buon_controllo_risparmio,
        'buon_controllo_spreco': mask_buon_controllo_spreco,
        'cattivo_controllo_spreco': mask_cattivo_controllo_spreco,
        'cattivo_controllo_risparmio': mask_cattivo_controllo_risparmio,
    }

    for i in range(12):
        start = i * STEPS_PER_MESE
        end = (i + 1) * STEPS_PER_MESE
        if start >= n: break
        
        steps_in_month = len(df_rl.iloc[start:end])
        if steps_in_month == 0: continue
        
        mese_str = f"Mese {i+1}"
        risultati_mensili[mese_str] = {}

        # Calcolo per i 4 quadranti RL
        for name, mask in all_masks.items():
            count = int(mask.iloc[start:end].sum())
            risultati_mensili[mese_str][name] = {
                'count': count,
                'percentage': round((count / steps_in_month) * 100, 2)
            }
        
        # Calcolo per il PID e risparmio energetico
        risparmio_mese = (Ppid.iloc[start:end] - Prl.iloc[start:end]).sum()
        pid_bad_steps_count = int(pessimi_step_mask_pid.iloc[start:end].sum())
        
        risultati_mensili[mese_str]['energy_saving_W'] = round(risparmio_mese, 2)
        risultati_mensili[mese_str]['pid_bad_steps'] = {
            'count': pid_bad_steps_count,
            'percentage': round((pid_bad_steps_count / steps_in_month) * 100, 2)
        }

    # --- 4. Stampa e Restituzione dei Risultati ---
    print("\n" + "=" * 80)
    print(f"Analisi Performance Comparativa - Modello RL: {model_name}")
    print("=" * 80)
    print(f"Numero totale di step analizzati: {n}")
    
    print("\n--- Analisi Energetica Complessiva ---")
    print(f"Risparmio Energetico Totale (P_pid - P_rl): {risparmio_energetico_totale:,.2f} W")
    print("\nDistribuzione Mensile del Risparmio Energetico:")
    print(f"{'Mese':<10} | {'Risparmio (P_pid - P_rl) [W]':<30}")
    print("-" * 45)
    for mese, dati in risultati_mensili.items():
        print(f"{mese:<10} | {dati['energy_saving_W']:>30,.2f}")

    print("\n" + "="*15 + " Riepilogo Comportamento Agente RL (% sul totale del mese) " + "="*15)
    header = f"{'Mese':<8} | {'Buon Controllo':<38} | {'Cattivo Controllo':<40}"
    sub_header = f"{'':<8} | {'+Risparmio':>18} {'+Spreco':>18} | {'+Spreco':>18} {'+Risparmio':>18}"
    print(header)
    print(sub_header)
    print("-" * len(header))
    
    for mese, dati in risultati_mensili.items():
        bc_r = dati['buon_controllo_risparmio']['percentage']
        bc_s = dati['buon_controllo_spreco']['percentage']
        cc_s = dati['cattivo_controllo_spreco']['percentage']
        cc_r = dati['cattivo_controllo_risparmio']['percentage']
        print(f"{mese:<8} | {bc_r:>17.2f}% {bc_s:>18.2f}% | {cc_s:>17.2f}% {cc_r:>18.2f}%")
        
    print("\nDefinizioni dei quadranti:")
    print("  - Buon Controllo + Risparmio: |dT| < 0.5  e  Prl < Ppid  (Caso Ideale)")
    print("  - Buon Controllo + Spreco:    |dT| < 0.5  e  Prl > Ppid")
    print("  - Cattivo Controllo + Spreco: |dT| > 0.5  e  Prl > Ppid  (Caso Peggiore)")
    print("  - Cattivo Controllo + Risparmio:|dT| > 0.5  e  Prl < Ppid")
    
    print("\n--- Confronto 'Pessimi Step' Totali ---")
    print(f"Totale 'Pessimi Step' RL (Cattivo Controllo + Spreco): {all_masks['cattivo_controllo_spreco'].sum()} ({all_masks['cattivo_controllo_spreco'].sum()/n*100:.2f}%)")
    print(f"Totale 'Pessimi Step' PID (|dT| > 0.5):               {numero_pessimi_step_pid} ({numero_pessimi_step_pid/n*100:.2f}%)")
    print("-" * 80)

    # Dizionario finale
    risultati = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'total_steps_analyzed': n,
        'total_energy_saving_W': round(risparmio_energetico_totale, 2),
        'total_bad_steps': {
            'rl_worst_case': {
                'count': int(all_masks['cattivo_controllo_spreco'].sum()),
                'percentage': round(all_masks['cattivo_controllo_spreco'].sum() / n * 100, 2)
            },
            'pid': {
                'count': numero_pessimi_step_pid,
                'percentage': round(numero_pessimi_step_pid / n * 100, 2)
            }
        },
        'monthly_analysis': risultati_mensili
    }
    return risultati
    
def days_to_result():
    algorithms = ['DDPG', 'TD3', 'SAC']
    agent_labels = ['1', '2', '3', '4', '5']

    # Dati per ogni agente
    ddpg = [175, 190, 194, 178, 163]
    td3 = [207, 207, 195, 160, 201]
    sac  = [201, 194, 198, 201, 201]

    # Liste dei valori
    data = [ddpg, td3, sac]
    means = [np.mean(d) for d in data]

    # Parametri del grafico
    bar_width = 0.2
    x = np.arange(len(agent_labels))  # posizione base per le barre

    # Offset per ogni algoritmo
    offsets = [-bar_width, 0, bar_width]

    # Colori
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Creazione della figura
    plt.figure(figsize=(10, 6))

    # Disegna le barre per ogni algoritmo
    for i, (alg, vals) in enumerate(zip(algorithms, data)):
        bars = plt.bar(x + offsets[i], vals, width=bar_width, label=f'{alg} (avg: {means[i]:.1f})', color=colors[i])

    # Plot global mean lines (full width)
    for i, mean in enumerate(means):
        plt.axhline(mean, color=colors[i], linestyle='--', linewidth=1)

    # Axis and labels
    plt.xticks(x, agent_labels)
    plt.xlabel('Agent')
    plt.ylabel('Day when stable and efficient control started')
    plt.ylim(150, 215)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("./graphs/days_to_result.png")

    plt.show()
    
def boxplot():
    # Trasformazione in DataFrame long
    records = []
    for (period, algo), values in ONLINE_RESULTS.items():
        for metric, val_list in values.items():
            for v in val_list:
                records.append({
                    'Training lenght': period,
                    'Algorithm': algo,
                    'Tipo': metric,
                    'Valore': v
                })

    df = pd.DataFrame(records)

    # Separazione dei plot
    sns.set(style="whitegrid")

    # BOXPLOT energia
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df['Tipo'] == 'P'], x='Training lenght', y='Valore', hue='Algorithm', whis=[0, 100])
    plt.axhline(1766.284, color='red', linestyle='--', label='PID')
    plt.ylabel('Energy [W]')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Algorithm', fontsize=14)
    plt.tight_layout()
    plt.savefig()

    # BOXPLOT comfort termico
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df['Tipo'] == 'ΔT'], x='Training lenght', y='Valore', hue='Algorithm', whis=[0, 100])
    plt.ylim(0, 6)
    plt.axhline(0.328, color='red', linestyle='--', label='PID')
    plt.ylabel('ΔT [K]')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Algorithm', fontsize=14)
    plt.tight_layout()
    plt.savefig()
    plt.show()
    
def plot_means_on():
    colors = {'DDPG': 'blue', 'TD3': 'green', 'SAC': 'red'}
    markers = {'1w': 'o', '2w': 's', '1m': 'D', '2m': '^', '3m': 'X'}

    plt.figure(figsize=(10, 7))

    for (duration, agent), metrics in OFF_ON_RESULTS.items():
        mean_p = np.mean(metrics['P'])
        mean_dt = np.mean(metrics['ΔT'])

        # Skip points outside the plotting range
        if mean_p > 2000 or mean_dt > 3:
            continue

        plt.scatter(
            mean_p,
            mean_dt,
            color=colors.get(agent, 'black'),
            marker=markers.get(duration, 'o'),
            s=50
        )

    # PID reference point
    plt.scatter(1766.284, 0.328, color='black', marker='*', s=150, label='PID')
    plt.text(1766.284, 0.328 - 0.05, 'PID', ha='center', va='top', fontsize=10)

    plt.xlabel("Average Energy [kW]")
    plt.ylabel("Average ΔT Error [°C]")
    plt.xlim(1400, 1900)
    plt.ylim(0, 2)
    plt.grid(True)

    # Custom legends
    color_legend = [
        Line2D([0], [0], marker='o', color='w', label='DDPG',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='TD3',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SAC',
               markerfacecolor='red', markersize=10),
    ]

    marker_legend = [
        Line2D([0], [0], marker=markers['1w'], color='k', label='1 week', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['2w'], color='k', label='2 weeks', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['1m'], color='k', label='1 month', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['2m'], color='k', label='2 months', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['3m'], color='k', label='3 months', linestyle='None', markersize=10),
    ]

    # Combine legends side by side in the top right
    legend1 = plt.legend(handles=color_legend, title="Algorithm", loc='upper right', bbox_to_anchor=(1, 1))
    plt.gca().add_artist(legend1)
    plt.legend(handles=marker_legend, title="Training Lenght", loc='upper right', bbox_to_anchor=(1, 0.67))

    plt.tight_layout()
    plt.savefig()
    plt.show()
    
def plot_means_off():
    colors = {'DDPG': 'blue', 'TD3': 'green', 'SAC': 'red'}
    markers = {'1m': 'o', '2m': 's', '3m': 'D', '6m': '^', '12m': 'X'}

    plt.figure(figsize=(10, 7))

    for (duration, agent), metrics in OFFLINE_RESULTS.items():
        mean_p = np.mean(metrics['P'])
        mean_dt = np.mean(metrics['ΔT'])

        plt.scatter(
            mean_p,
            mean_dt,
            color=colors.get(agent, 'black'),
            marker=markers.get(duration, 'o'),
            s=50
        )

    # PID reference point
    plt.scatter(1766.284, 0.328, color='black', marker='*', s=150, label='PID')
    plt.text(1766.284, 0.328 - 0.05, 'PID', ha='center', va='top', fontsize=10)

    plt.xlabel("Average Energy [kW]")
    plt.ylabel("Average ΔT Error [°C]")
    plt.xlim(1400, 2500)
    plt.ylim(0, 5)
    plt.grid(True)

    # Custom legends
    color_legend = [
        Line2D([0], [0], marker='o', color='w', label='DDPG',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='TD3',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SAC',
               markerfacecolor='red', markersize=10),
    ]

    marker_legend = [
        Line2D([0], [0], marker=markers['1m'], color='k', label='1 month', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['2m'], color='k', label='2 months', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['3m'], color='k', label='3 months', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['6m'], color='k', label='6 months', linestyle='None', markersize=10),
        Line2D([0], [0], marker=markers['12m'], color='k', label='12 months', linestyle='None', markersize=10),
    ]

    # Combine legends side by side in the top right
    legend1 = plt.legend(handles=color_legend, title="Algorithm", loc='upper left')
    plt.gca().add_artist(legend1)
    plt.legend(handles=marker_legend, title="Dataset Lenght", loc='upper left', bbox_to_anchor=(0.12, 1))

    plt.tight_layout()
    plt.savefig()
    plt.show()
    
def plot_performance_complessiva(dati_rl, dati_pid):
    """
    Crea un istogramma verticale con le performance complessive dei modelli RL
    e aggiunge una linea per la performance del PID come riferimento.
    """
    # --- 1. Preparazione dei Dati ---
    modelli = list(dati_rl.keys())
    rl_percentages = list(dati_rl.values())
    
    # Associa il corretto valore PID a ciascun modello RL
    # Questo è necessario perché i dati del PID sono diversi per i due esperimenti
    pid_percentages = [
        dati_pid['PID (vs SAC/DDPG)'],  # PID per SAC
        dati_pid['PID (vs SAC/DDPG)'],  # PID per DDPG
        dati_pid['PID (vs TD3)']        # PID per TD3
    ]

    # --- 2. Impostazione del Grafico ---
    x = np.arange(len(modelli))  # Posizioni delle etichette per i gruppi (SAC, DDPG, TD3)
    width = 0.35  # Larghezza delle barre (più stretta del default)

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- 3. Creazione delle Barre Raggruppate ---
    # Barre per i modelli RL, posizionate a sinistra del centro del gruppo
    rects1 = ax.bar(x - width/2, rl_percentages, width, label='RL Model', color='#1f77b4')
    
    # Barre per il PID, posizionate a destra del centro del gruppo
    rects2 = ax.bar(x + width/2, pid_percentages, width, label='PID Baseline', color='#d62728')

    # --- 4. Aggiunta di Titoli ed Etichette in Inglese ---
    ax.set_ylabel('Percentage of "Bad Steps" (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(modelli)
    ax.legend()
    
    # Aggiungi etichette con il valore numerico sopra ogni barra per chiarezza
    ax.bar_label(rects1, padding=3, fmt='%.2f%%')
    ax.bar_label(rects2, padding=3, fmt='%.2f%%')

    # Migliora il layout per evitare sovrapposizioni
    fig.tight_layout()
    plt.savefig()
    plt.show()
    
# Funzione 2: Plotta i dati mensili di un singolo algoritmo vs PID
def plot_performance_mensile(monthly_data):
    """
    Creates a grouped bar chart to compare the monthly performance
    of an RL model against the PID baseline.

    Args:
        monthly_data (dict): A dictionary containing model name, months, 
                             and performance percentages for both RL and PID.
    """
    # --- 1. Estrazione e traduzione dei dati ---
    model_name = monthly_data["nome_modello"]
    
    # Traduce le etichette dei mesi da "Mese X" a "Month X"
    months_italian = monthly_data["mesi"]
    months_english = [m.replace("Mese", "Month") for m in months_italian]
    
    rl_values = monthly_data["rl_percentuali"]
    pid_values = monthly_data["pid_percentuali"]
    
    # --- 2. Impostazione del Grafico ---
    x = np.arange(len(months_english))  # The label locations for the months
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- 3. Creazione delle Barre Raggruppate ---
    rects1 = ax.bar(x - width/2, rl_values, width, label=f'RL ({model_name})', color=monthly_data["color"])
    rects2 = ax.bar(x + width/2, pid_values, width, label='PID Baseline', color='#d62728')

    # --- 4. Aggiunta di Titoli ed Etichette in Inglese ---
    ax.set_ylabel('Percentage of "Bad Steps" (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(months_english)
    ax.legend()

    # Aggiunge etichette con il valore numerico e il simbolo '%' sopra ogni barra
    ax.bar_label(rects1, padding=3, fmt='%.2f%%')
    ax.bar_label(rects2, padding=3, fmt='%.2f%%')

    fig.tight_layout()
    plt.savefig()
    plt.show()
    
def plot_performance_confronto(lista_dati_agenti, nome_file_salvataggio="performance_confronto.png"):
    """
    Crea un grafico a barre raggruppate per confrontare la performance 
    di più modelli RL con una singola linea di base PID.

    Args:
        lista_dati_agenti (list): Una lista di dizionari. Ogni dizionario 
                                  contiene i dati per un agente (nome, etichette temporali, 
                                  percentuali RL e percentuali PID).
        nome_file_salvataggio (str): Il nome del file con cui salvare il grafico.
    """
    color_map = {
        "DDPG": "#4c72b0",  # Blu scuro
        "TD3":  "#dd8452",  # Arancione/Terracotta
        "SAC":  "#55a868",  # Verde
        "PID":  "#c44e52"   # Rosso per la baseline
    }
    
    # --- 1. Estrazione e preparazione dei dati ---
    if not lista_dati_agenti:
        print("La lista dei dati degli agenti è vuota.")
        return

    # Usa il primo agente per le etichette e i dati PID di riferimento
    dati_riferimento = lista_dati_agenti[0]
    
    # Determina se i dati sono mensili o settimanali
    if "settimane" in dati_riferimento:
        etichette_italiane = dati_riferimento["settimane"]
        etichette_inglesi = [e.replace("Settimana", "Week") for e in etichette_italiane]
        unita_tempo = "Weekly"
    elif "mesi" in dati_riferimento:
        etichette_italiane = dati_riferimento["mesi"]
        etichette_inglesi = [e.replace("Mese", "Month") for e in etichette_italiane]
        unita_tempo = "Monthly"
    else:
        raise ValueError("I dizionari devono contenere la chiave 'mesi' o 'settimane'.")

    # Prende i valori PID dal primo agente come baseline per il grafico
    # NOTA: Questa è un'assunzione. Se ogni agente avesse una baseline molto diversa,
    # potrebbe essere necessario un approccio differente.
    pid_values = dati_riferimento["pid_percentuali"]
    
    num_agenti = len(lista_dati_agenti)
    
    # --- 2. Impostazione del Grafico ---
    x = np.arange(len(etichette_inglesi))  # Posizioni per i gruppi di barre
    width = 0.8 / (num_agenti + 1)  # Larghezza delle barre, calcolata per adattarsi
    
    fig, ax = plt.subplots(figsize=(12, 6)) # Grafico più largo per contenere più dati

    # --- 3. Creazione delle Barre Raggruppate ---
    all_rects = []
    for i, dati_agente in enumerate(lista_dati_agenti):
        model_name = dati_agente["nome_modello"]
        rl_values = dati_agente["rl_percentuali"]
        offset = width * (i - num_agenti / 2)
        
        rects = ax.bar(x + offset, rl_values, width, label=f'{model_name}', color=color_map[model_name])
        all_rects.append(rects)

    # --- 4. Aggiunta di Titoli ed Etichette ---
    ax.set_ylabel('Percentage of "Bad Steps" (%)')
    ax.set_xticks(x)
    # Ruota le etichette dell'asse x se sono troppe per essere lette comodamente
    if len(etichette_inglesi) > 8:
        ax.set_xticklabels(etichette_inglesi, rotation=45, ha="right")
    else:
        ax.set_xticklabels(etichette_inglesi)
        
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Aggiunge etichette con i valori sopra ogni barra
    for rects in all_rects:
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=8, rotation=90)

    fig.tight_layout()
    plt.savefig()
    plt.show()
    
def plot_behavior_comparison(group_data):
    """
    Crea un grafico a barre orizzontali raggruppate e impilate.
    """
    n_agents = len(group_data)
    if n_agents == 0:
        return

    period_labels = [d['period_label'] for d in group_data[0]['time_series_data']]
    n_periods = len(period_labels)
    agent_names = [d['model_name'].split(' ')[0] for d in group_data]

    # --- Definizione di colori e etichette ---
    quadrant_keys = ['good_control_saving', 'good_control_waste', 'bad_control_saving', 'bad_control_waste']
    quadrant_labels = {
        'good_control_saving': 'Good Control, Saving (Ideal)',
        'good_control_waste': 'Good Control, Waste',
        'bad_control_saving': 'Bad Control, Saving',
        'bad_control_waste': 'Bad Control, Waste (Worst)'
    }
    colors = {
        'good_control_saving': '#2ca02c', 'good_control_waste': '#98df8a',
        'bad_control_saving': '#ff7f0e', 'bad_control_waste': '#d62728'
    }
    
    # --- Logica di posizionamento per barre orizzontali ---
    bar_height = 0.8
    bar_gap = 0.1
    bar_spacing = bar_height + bar_gap
    group_height = bar_spacing * n_agents
    group_spacing = group_height + bar_height * 2

    # Posizioni centrali per ogni gruppo sull'asse Y
    y_main_positions = np.arange(n_periods) * group_spacing
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Loop per disegnare ogni barra di ogni agente ---
    for i, agent_data in enumerate(group_data):
        offset = (i - (n_agents - 1) / 2) * bar_spacing
        agent_y_positions = y_main_positions + offset
        
        # Posiziona i nomi degli agenti a sinistra delle barre
        for pos_y in agent_y_positions:
            ax.text(x=-2, y=pos_y, s=agent_names[i], ha='right', va='center', fontsize=11)
        
        left = np.zeros(n_periods)
        for key in quadrant_keys:
            values = np.array([d['quadrants'][key] for d in agent_data['time_series_data']])
            # Modifica: Utilizzo di ax.barh per barre orizzontali
            ax.barh(agent_y_positions, values, height=bar_height, left=left, color=colors[key])
            left += values

    # --- Formattazione degli assi ---
    # Asse Y (Periodi)
    ax.set_yticks(y_main_positions)
    ax.set_yticklabels(period_labels, fontsize=14, weight='bold')
    ax.tick_params(axis='y', which='major', pad=80, length=0, labelsize=14) # Aumenta pad per dare spazio ai nomi degli agenti
    
    # Asse X (Percentuale)
    ax.set_xlabel('Percentage of Time (%)', fontsize=14)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.invert_yaxis() # Mette il primo periodo in alto

    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[key], label=quadrant_labels[key]) for key in quadrant_keys]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    fig.tight_layout(rect=[0.05, 0, 1, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig()
    plt.show()
    
def plot_energy_saving_comparison(group_data):
    n_agents = len(group_data)
    if n_agents == 0:
        return

    period_labels = [d['period_label'] for d in group_data[0]['time_series_data']]
    n_periods = len(period_labels)
    agent_names = [d['model_name'].split(' ')[0] for d in group_data]

    bar_height = 0.8
    bar_gap = 0.1
    bar_spacing = bar_height + bar_gap
    group_height = bar_spacing * n_agents
    group_spacing = group_height + bar_height * 2
    y_main_positions = np.arange(n_periods) * group_spacing
    
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, agent_data in enumerate(group_data):
        offset = (i - (n_agents - 1) / 2) * bar_spacing
        agent_y_positions = y_main_positions + offset
        
        energy_values_kw = [d['energy_saving_W'] / 1000 for d in agent_data['time_series_data']]
        
        for y_pos, value_kw in zip(agent_y_positions, energy_values_kw):
            # Disegna la barra energetica
            color = '#2ca02c' if value_kw >= 0 else '#d62728'
            ax.barh(y_pos, value_kw, height=bar_height, color=color)
            
            # Scrive il valore numerico sulla barra
            if abs(value_kw) > 0:
                x_pos_text = value_kw - np.sign(value_kw) * 2
                ha_val = 'right' if value_kw > 0 else 'left'
                ax.text(x_pos_text, y_pos, f'{value_kw:,.0f}', 
                        ha=ha_val, va='center',
                        fontsize=10, color='white', weight='bold')
            
            # Logica per posizionare l'etichetta dell'algoritmo
            if value_kw >= 0: # Valore positivo, barra a destra -> etichetta a sinistra
                label_x_pos = -1
                ha_label = 'right'
            else: # Valore negativo, barra a sinistra -> etichetta a destra
                label_x_pos = 1
                ha_label = 'left'
            
            ax.text(label_x_pos, y_pos, agent_names[i], 
                    ha=ha_label, va='center', fontsize=11)

    ax.axvline(0, color='black', linewidth=1.5, linestyle='-')

    ax.tick_params(axis='y', which='major', pad=15, length=0)
    ax.set_yticks(y_main_positions)
    ax.set_yticklabels(period_labels)
    
    ax.set_xlabel('Energy Saving (vs PID) [kW]')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xscale('symlog', linthresh=10)
    ax.invert_yaxis()

    green_patch = mpatches.Patch(color='#2ca02c', label='Energy Saving (Gain)')
    red_patch = mpatches.Patch(color='#d62728', label='Energy Waste (Loss)')
    ax.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=11)

    fig.tight_layout(rect=[0.05, 0, 1, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig()
    plt.show()
    
def calculate_delta_percentage_in_chunks(file_path: str):
    """
    Legge un file Excel, lo divide in blocchi e calcola la percentuale
    di valori > 0.5 nella colonna 'delta' per ciascun blocco.

    Args:
        file_path (str): Il percorso del file Excel da analizzare.
    """
    # La dimensione del blocco standard.
    chunk_size = 2016

    try:
        # Legge l'intero file Excel in un DataFrame di pandas.
        df = pd.read_excel(file_path)

        # Controlla se la colonna 'delta' esiste.
        if 'delta' not in df.columns:
            print(f"Errore: La colonna 'delta' non è stata trovata nel file '{file_path}'.")
            return

        print(f"Analisi del file: '{file_path}' in corso...")
        
        # Itera attraverso il DataFrame a passi di 'chunk_size'.
        chunk_number = 1
        for start_index in range(0, len(df), chunk_size):
            end_index = start_index + chunk_size
            
            # Estrae il blocco di dati corrente.
            # .iloc gestisce correttamente l'ultimo blocco anche se è più piccolo.
            chunk = df.iloc[start_index:end_index]
            
            # Se il blocco è vuoto, salta al prossimo.
            if chunk.empty:
                continue

            # Calcola il numero di casi in cui 'delta' > 0.5.
            # Il confronto (chunk['delta'] > 0.5) crea una serie di True/False.
            # .sum() tratta True come 1 e False come 0, contando i casi positivi.
            count_over_threshold = (abs(chunk['delta']) > 0.5).sum()
            
            # Calcola la dimensione totale del blocco corrente.
            total_in_chunk = len(chunk)
            
            # Calcola la percentuale, gestendo il caso di un blocco vuoto (divisione per zero).
            if total_in_chunk > 0:
                percentage = (count_over_threshold / total_in_chunk) * 100
            else:
                percentage = 0.0

            # Stampa il risultato per il blocco corrente.
            print(f"Blocco {chunk_number} (righe {start_index+1}-{start_index+total_in_chunk}):")
            print(f"  - La percentuale di valori delta > 0.5 è: {percentage:.2f}%")
            
            chunk_number += 1

    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore imprevisto: {e}")    


if __name__ == "__main__":
    # reward_convergence(["ddpg", "sac", "td3"], DAY, HALF_YEAR*2)
    # avg_reward_convergence('', DAY, MONTH*6, start_index=20, end_index=24)
    # saving_momentum("sac_21_23")
    # days_to_result()
    # boxplot()
    # plot_energy_vs_temp_delta(ONLINE_RESULTS)
    # plot_means_on()
    # export_daily_avg_rewards()
    # plot_avg_daily_rewards()
    # training_analysis()
    
    # analizza_performance_agente("sac_off_on_1m_feb", "PID_dataset_aosta_feb_test")
    
    # plot_performance_complessiva(dati_complessivi_rl, dati_complessivi_pid)
    # plot_performance_mensile(dati_mensili_td3)
    # plot_performance_mensile(dati_settimanali_td3_off_on)
    # plot_performance_mensile(dati_settimanali_sac_off_on)
    
    # plot_performance_confronto([dati_mensili_ddpg, dati_mensili_td3, dati_mensili_sac], nome_file_salvataggio="performance_mensile_confronto.png")
    
    # plot_behavior_comparison(off_on_data)
    # plot_energy_saving_comparison(on_data)
    
    calculate_delta_percentage_in_chunks()