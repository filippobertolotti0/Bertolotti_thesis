import gymnasium as gym
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
from tqdm import tqdm
import registration
import matplotlib.pyplot as plt
import pandas as pd
from utils import normalize, unnormalize, DAY, WEEK, MONTH, HALF_YEAR, YEAR
import torch
import energym

if __name__ == "__main__":
    env = energym.make('SimpleHouseRad-v0', weather="aosta2024", start_day=10, start_month=10, year=2024, simulation_days=180, eval_mode=False)
    tout = []
    tin = []
    for i in tqdm(range(DAY)):
        env.step({'u': [0.1]})
        out = env.get_output()
        tout.append(out["TOut.T"])
        tin.append(out["temRoo.T"])
    print(max(tout))
    print(min(tout))
    print(max(tin))
    print(min(tin))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(tin)
    plt.show()
    
###################### EPW FILES CORRECTION ######################
# import re

# # Percorso al file EPW
# epw_file = "./epw_files/aosta/aosta2019.epw"
# corrected_file = "./epw_files/aosta/aosta2019.epw"

# # Leggi il file EPW
# with open(epw_file, "r") as file:
#     lines = file.readlines()
# # Correggi i valori "99" nella colonna specifica
# corrected_lines = []
# for line in lines:
#     # Verifica se la riga è una riga di dati (esclude header e commenti)
#     if re.match(r"^\d{4},", line):
#         # Suddividi i valori
#         values = line.split(",")
#         # Colonna 24: "Opaque sky cover at indicated time"
#         if values[23].strip() == "99":
#             values[23] = "5"
#         if values[25].strip() == "99999":
#             values[25] = "3"  # Valore sostitutivo (ad esempio, copertura media)
#         corrected_lines.append(",".join(values))
#     else:
#         corrected_lines.append(line)

# # Scrivi il file corretto
# with open(corrected_file, "w") as file:
#     file.writelines(corrected_lines)

# print(f"File corretto salvato come {corrected_file}")

################## MOS FILE CORRECTION ##################
# File di input e output
# input_file = "../../../epw_files/aosta/aosta2024.mos"  # Sostituisci con il nome del tuo file
# output_file = "../../../epw_files/aosta/aosta2024.mos"

# # Leggere il file riga per riga
# with open(input_file, "r") as file:
#     lines = file.readlines()

# # Trova la linea che separa l'header dai dati
# data_start_index = next(
#     i for i, line in enumerate(lines) 
#     if not (line.startswith("#") or (i == 1 and line.startswith("double")))
# )

# # Dividi header e dati
# header = lines[:data_start_index]
# data = lines[data_start_index:]

# # Funzione per modificare il timestep
# timestep = 0.0
# def modify_timestep(line, timestep):
#     # Dividi la riga in colonne
#     values = line.strip().split()
#     # Modifica il valore del timestep (prima colonna)
#     values[0] = str(timestep)  # Ad esempio, raddoppia il timestep
#     # Ricostruisci la riga modificata
#     return "\t".join(values)

# # Modifica i dati riga per riga
# modified_data = []
# for line in data:
#     modified_data.append(modify_timestep(line, timestep))
#     timestep += 3600.0
    
# # Scrivi il file modificato
# with open(output_file, "w") as file:
#     file.writelines(header)
#     file.write("\n".join(modified_data) + "\n")