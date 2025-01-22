def read_first_column(file_path, skip_lines=40):
    """
    Legge solo il primo elemento di ogni riga di una tabella, ignorando le prime `skip_lines` righe.
    I dati possono essere separati da tabulazioni o spazi multipli.
    """
    with open(file_path, "r") as file:
        # Salta le prime `skip_lines` righe
        for _ in range(skip_lines):
            next(file)
        
        # Estrai solo il primo elemento di ogni riga
        first_column = [line.strip().split()[0] for line in file if line.strip()]
    
    return first_column

# Percorsi dei file
file_path_basel = "Basel_Fixed.mos"
file_path_cherasco = "turin2020.mos"

# Estrazione della prima colonna
first_column_basel = read_first_column(file_path_basel)
first_column_cherasco = read_first_column(file_path_cherasco)

# Stampa delle prime 5 righe della prima colonna per ciascun file per verifica
print("Prima colonna - Basel_Fixed:")
print(first_column_basel[:5])
print(len(first_column_basel))

print("\nPrima colonna - Cherasco2020:")
print(first_column_cherasco[:5])
print(len(first_column_cherasco))

for i in range(1, 8760):
    delta_basel = float(first_column_basel[i]) - float(first_column_basel[i-1])
    delta_cherasco = float(first_column_cherasco[i]) - float(first_column_cherasco[i-1])
    if delta_basel != 3600.0:
        print(f"Errore a Basel alla riga {i}: {delta_basel}")
    if delta_cherasco != 3600.0:
        print(f"Errore a Cherasco alla riga {i}: {delta_cherasco}")
    # print(f"{float(first_column_basel[i]) - float(first_column_basel[i-1])} - {float(first_column_cherasco[i]) - float(first_column_cherasco[i-1])}")
    # if first_column_basel[i] != first_column_cherasco[i]:
    #     print(f"Errore alla riga {i}: {first_column_basel[i]} != {first_column_cherasco[i]}")
    #     break