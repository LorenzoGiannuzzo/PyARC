
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from itertools import product


class DataPreprocessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_negative_values(self):
        if "Consumption" in self.dataframe.columns:
            self.dataframe["Consumption"] = np.where(self.dataframe["Consumption"] < 0, np.nan, self.dataframe["Consumption"])
            return self.dataframe
        else:
            print("The 'Consumption' column is not present in the dataframe.")
            return None

    @staticmethod
    def replace_max_daily_zero_consumption(dataframe):
        max_zero_mask = (dataframe["Consumption"] == 0) & ~dataframe["Consumption"].isna()
        max_zero_profiles = dataframe[max_zero_mask].groupby(["User", "Year", "Month", "Day"])["Consumption"].idxmax()
        dataframe.loc[max_zero_profiles, "Consumption"] = np.nan
        return dataframe

    @staticmethod
    def interpolate_missing_values(df, max_gap=3):
        df = df.sort_values(by=["User", "Year", "Month", "Day"])
        df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce')

        grouped_df = df.groupby("User")

        def interpolate_group(group):
            group['Consumption'] = group['Consumption'].interpolate(method='linear', limit=max_gap)
            return group

        interpolated_df = grouped_df.apply(interpolate_group)
        interpolated_df.reset_index(drop=True, inplace=True)

        return interpolated_df

    @staticmethod
    def fill_missing_values_with_monthly_mean(df):
        df = df.sort_values(by=["User", "Year", "Month", "Day"])
        monthly_means = df.groupby(["User", "Year", "Month"])["Consumption"].mean().reset_index()
        df_filled = pd.merge(df, monthly_means, on=["User", "Year", "Month"], how="left", suffixes=('', '_mean'))
        df_filled["Consumption"] = df_filled["Consumption"].fillna(df_filled["Consumption_mean"])
        df_filled = df_filled.drop(columns=["Consumption_mean"])
        return df_filled

    @staticmethod
    def remove_outliers_iqr(df):
        column_name = "Consumption"
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[column_name] = df[column_name].apply(lambda x: x if lower_bound <= x <= upper_bound else None)
        return df

    @staticmethod
    def infrequent_profiles(df):
        # Creazione del DataFrame risultato
        df_result = pd.DataFrame()

        # Iterazione su ogni mese
        for month in df['Month'].unique():
            # Filtraggio del DataFrame per il mese specifico
            df_month = df[df['Month'] == month]

            # Raggruppamento per "Year", "Month", e "Day" per ottenere il profilo giornaliero
            df_grouped = df_month.groupby(['Year', 'Month', 'Day']).agg({
                'User': 'nunique',
                'Norm_consumption': list
            }).reset_index()

            # Converte la lista di consumi in colonne separate
            consumption_columns = pd.DataFrame(df_grouped['Norm_consumption'].tolist(), index=df_grouped.index)
            consumption_columns.columns = ['Hour_{}'.format(i) for i in range(24)]
            df_grouped = pd.concat([df_grouped, consumption_columns], axis=1)

            # Seleziona solo le colonne dei consumi orari per il clustering
            features = df_grouped.iloc[:, 3:]

            # Prova diverse configurazioni di numero di cluster e calcola l'indice di Davies-Bouldin
            best_davies_bouldin = float('inf')
            best_num_clusters = 0

            for num_clusters in range(7, 21):
                kmeans = KMeans(n_clusters=num_clusters, algorithm='lloyd', random_state=42)
                labels = kmeans.fit_predict(features)
                db_score = davies_bouldin_score(features, labels)

                if db_score < best_davies_bouldin:
                    best_davies_bouldin = db_score
                    best_num_clusters = num_clusters

            # Esegui il clustering con il numero ottimale di cluster
            kmeans = KMeans(n_clusters=best_num_clusters, algorithm='lloyd', random_state=42)
            df_grouped['Cluster'] = kmeans.fit_predict(features)

            # Controlla il numero di utenti e profili giornalieri nei cluster e filtra
            cluster_counts = df_grouped['Cluster'].value_counts(normalize=True)
            valid_clusters = cluster_counts[cluster_counts >= 0.05].index.tolist()
            df_grouped_filtered = df_grouped[df_grouped['Cluster'].isin(valid_clusters)]

            # Aggiungi il risultato al DataFrame risultato
            df_result = pd.concat([df_result, df_grouped_filtered])

        # Rimuovi la colonna 'Cluster' dal DataFrame risultato
        df_result = df_result.drop(columns=['Cluster'])

        return df_result