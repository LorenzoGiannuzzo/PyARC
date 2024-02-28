import os
import pandas as pd
import numpy as np
#
# def elimina_80_percento_righe(df):
#     # Trova l'elenco degli utenti unici
#     utenti_unici = df['User'].unique()
#
#     # Per ciascun utente, elimina casualmente l'80% delle righe
#     #for utente in utenti_unici:
#     righe_utente = df
#     righe_da_elim =  int(0.95 * len(righe_utente[,1]))
#     righe_eliminate = np.random.choice(righe_utente.index, righe_da_elim, replace=False)
#     df = df.drop(righe_eliminate)
#
#     return df

def remove_90_percent_rows(df):
    # Find the list of unique users
    unique_users = df['User'].unique()

    # Calculate the number of users to keep (10%)
    num_users_to_keep = int(0.01 * len(unique_users))

    # Randomly select 10% of users to keep
    users_to_keep = np.random.choice(unique_users, num_users_to_keep, replace=False)

    # Filter the DataFrame, keeping only the rows corresponding to selected users
    reduced_df = df[df['User'].isin(users_to_keep)]

    return reduced_df

# Percorso del file CSV
data_path = os.path.join("..", "test", "data.csv")

# Carica il dataframe da CSV
data = pd.read_csv(data_path)

# Chiamata alla funzione per eliminare casualmente l'80% delle righe per ciascun utente
data_modificato = remove_90_percent_rows(data)

# Percorso per esportare il dataframe modificato in formato CSV
output_path = os.path.join(os.path.dirname(__file__), "data_test2.csv")

# Esporta il dataframe modificato in formato CSV
data_modificato.to_csv(output_path, index=False)

# Ora il dataframe modificato è stato esportato nella stessa directory dello script con il nome 'data_modificato.csv'


