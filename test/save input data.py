
#after create permutation ratio in the training funciton
box3 = data
selected_columns = ["User", "Year", "Month", "F1", "F2", "F3"]
new_dataframe = box3[selected_columns]

# Elimina le righe duplicate
new_dataframe = new_dataframe.drop_duplicates()
cartella = os.path.join("..", "data", "Input Data")

# Specifica il percorso completo del file CSV
percorso_completo = os.path.join(cartella, 'data.csv')