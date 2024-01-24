import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

class DataPreprocessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_negative_values(self):
        # Check if the "Consumption" column exists in the dataframe
        if "Consumption" in self.dataframe.columns:
            # Replace negative values with NaN in the "Consumption" column
            self.dataframe["Consumption"] = np.where(self.dataframe["Consumption"] < 0, np.nan, self.dataframe["Consumption"])
            return self.dataframe
        else:
            print("The 'Consumption' column is not present in the dataframe.")
            return None

    import pandas as pd
    import numpy as np

    def replace_max_daily_zero_consumption(dataframe):

        # Identify daily profiles with maximum zero value
        max_zero_mask = (dataframe["Consumption"] == 0) & ~dataframe["Consumption"].isna()
        max_zero_profiles = dataframe[max_zero_mask].groupby(["User", "Year", "Month", "Day"])["Consumption"].idxmax()

        # Replace corresponding values with NaN
        dataframe.loc[max_zero_profiles, "Consumption"] = np.nan

        return dataframe

    def interpolate_missing_values(df, max_gap=3):
            # Ordina il DataFrame in base a "User", "Year", "Month" e "Day"
            df = df.sort_values(by=["User", "Year", "Month", "Day"])

            # Trasforma la colonna "Consumption" in valori numerici
            df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce')

            # Raggruppa il DataFrame per "User"
            grouped_df = df.groupby("User")

            # Funzione di interpolazione lineare per un gruppo di dati
            def interpolate_group(group):
                group['Consumption'] = group['Consumption'].interpolate(method='linear', limit=max_gap)
                return group

            # Applica la funzione di interpolazione lineare a ciascun gruppo
            interpolated_df = grouped_df.apply(interpolate_group)

            # Resetta l'indice del DataFrame risultante
            interpolated_df.reset_index(drop=True, inplace=True)

            return interpolated_df

    def fill_missing_values_with_monthly_mean(df):
        # Ordina il DataFrame in base a "User", "Year", "Month", "Day"
        df = df.sort_values(by=["User", "Year", "Month", "Day"])

        # Calcola la media mensile per ogni utente e anno
        monthly_means = df.groupby(["User", "Year", "Month"])["Consumption"].mean().reset_index()

        # Unisci la media mensile con il DataFrame originale per ottenere i valori mancanti
        df_filled = pd.merge(df, monthly_means, on=["User", "Year", "Month"], how="left", suffixes=('', '_mean'))

        # Riempi i valori mancanti con la media mensile
        df_filled["Consumption"] = df_filled["Consumption"].fillna(df_filled["Consumption_mean"])

        # Elimina le colonne temporanee utilizzate per il calcolo
        df_filled = df_filled.drop(columns=["Consumption_mean"])

        return df_filled

    def remove_outliers_iqr(df):
        # Specify the column name
        column_name = "Consumption"

        # Calculate the first and third quartiles
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)

        # Calculate the interquartile range (IQR)
        iqr = q3 - q1

        # Define the bounds for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Replace outliers with NaN
        df[column_name] = df[column_name].apply(lambda x: x if lower_bound <= x <= upper_bound else None)

        return df

    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score


    def infrequent_profiles(dataframe):
            # Filtraggio del dataframe per ogni "Dayname"
            unique_daynames = dataframe['Dayname'].unique()

            for dayname in unique_daynames:
                day_data = dataframe[dataframe['Dayname'] == dayname].copy()

                # Rimuovi righe con valori mancanti nella colonna 'Norm_consumption'
                day_data = day_data.dropna(subset=['Norm_consumption'])

                if not day_data.empty:
                    # Inizializzazione delle variabili per il numero di cluster e il valore minimo di Davies-Bouldin
                    best_num_clusters = None
                    min_davies_bouldin = float('inf')

                    # Iterazione attraverso il numero di cluster da 7 a 20
                    for num_clusters in range(7, 21):
                        # Esecuzione del clustering k-means
                        kmeans = KMeans(n_clusters=num_clusters, n_init=100, max_iter=700, algorithm="lloyd")
                        day_data['Cluster'] = kmeans.fit_predict(day_data[['Norm_consumption']])

                        # Calcolo dell'indice di Davies-Bouldin
                        davies_bouldin = davies_bouldin_score(day_data[['Norm_consumption']], day_data['Cluster'])

                        # Se l'indice di Davies-Bouldin è minore del valore minimo trovato finora, aggiorna i valori
                        if davies_bouldin < min_davies_bouldin:
                            min_davies_bouldin = davies_bouldin
                            best_num_clusters = num_clusters

                    # Esecuzione del clustering k-means con il numero ottimale di cluster
                    kmeans = KMeans(n_clusters=best_num_clusters, n_init=100, max_iter=700, algorithm="lloyd")
                    day_data['Cluster'] = kmeans.fit_predict(day_data[['Norm_consumption']])

                    # Conteggio del numero di profili giornalieri e utenti unici in ogni cluster
                    cluster_counts = day_data['Cluster'].value_counts()
                    unique_users_in_clusters = day_data.groupby('Cluster')['User'].nunique()

                    # Filtraggio dei cluster poco popolati
                    filtered_clusters = cluster_counts[
                        (cluster_counts >= 0.05 * len(day_data)) &
                        (unique_users_in_clusters >= 0.05 * len(day_data['User'].unique()))
                        ].index

                    # Applicazione del filtro al dataframe temporaneo
                    day_data = day_data[day_data['Cluster'].isin(filtered_clusters)]

                    # Aggiorna il DataFrame principale con il DataFrame temporaneo
                    dataframe.loc[dataframe['Dayname'] == dayname, 'Cluster'] = day_data['Cluster'].copy()

            return dataframe