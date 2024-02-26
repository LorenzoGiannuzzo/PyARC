import os
import pandas as pd
import numpy as np

def elimina_80_percento_righe(df):
    # Trova l'elenco degli utenti unici
    utenti_unici = df['User'].unique()

    # Per ciascun utente, elimina casualmente l'80% delle righe
    for utente in utenti_unici:
        righe_utente = df[df['User'] == utente]
        righe_da_elim = int(0.8 * len(righe_utente))
        righe_eliminate = np.random.choice(righe_utente.index, righe_da_elim, replace=False)
        df = df.drop(righe_eliminate)

    return df

# Percorso del file CSV
data_path = os.path.join("..", "data", "Input Data", "data.csv")

# Carica il dataframe da CSV
data = pd.read_csv(data_path)

# Chiamata alla funzione per eliminare casualmente l'80% delle righe per ciascun utente
data_modificato = elimina_80_percento_righe(data)

# Percorso per esportare il dataframe modificato in formato CSV
output_path = os.path.join(os.path.dirname(__file__), "data_modificato.csv")

# Esporta il dataframe modificato in formato CSV
data_modificato.to_csv(output_path, index=False)

# Ora il dataframe modificato è stato esportato nella stessa directory dello script con il nome 'data_modificato.csv'
