
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
    def filter_users(df):
        df = df.sort_values(by=["User", "Year", "Month", "Day"])
        # Converti la colonna "Hour" in valori numerici se non è già di tipo numerico
        if not pd.api.types.is_numeric_dtype(df['Hour']):
            df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')

        # Group by "User", "Year", "Month", "Day" and check if all hours from 0 to 23 are present
        valid_users_series = df.groupby(["User", "Year", "Month", "Day"])["Hour"].apply(
            lambda x: set(x) == set(range(24)))

        # Creazione di un nuovo DataFrame con le colonne "User", "Year", "Month", "Day", "Valid"
        valid_df = pd.DataFrame(valid_users_series).reset_index()
        valid_df.columns = ["User", "Year", "Month", "Day", "Valid"]

        # Filtraggio del DataFrame in base agli utenti validi
        filtered_df = df[df["User"].isin(valid_df[valid_df["Valid"]]["User"])]

        # Calcolo del numero di utenti eliminati
        num_users_eliminated = len(df["User"].unique()) - len(filtered_df["User"].unique())

        # Output message
        print(f"{num_users_eliminated} users eliminated. DataFrame processed successfully.")

        return filtered_df
    @staticmethod
    def monthly_average_consumption(dataframe):
        # Create a copy of the input DataFrame
        result_df = dataframe.copy()

        # Group the DataFrame by User, Year, and Month
        grouped_df = result_df.groupby(["User", "Year", "Month", "Hour"])

        # Calculate the monthly average normalized consumption
        result_df["M_consumption"] = grouped_df["Norm_consumption"].transform("mean")

        return result_df

    @staticmethod
    def reshape_dataframe(input_df):
        # Raggruppa il dataframe per le colonne 'User', 'Year', 'Month'
        grouped_df = input_df.groupby(['User', 'Year', 'Month','Hour'])['M_consumption'].mean().reset_index()

        return grouped_df


    @staticmethod
    def infrequent_profiles(df):
        # Inizializza il dataframe risultante
        result_df = pd.DataFrame()

        # Itera per ogni giorno della settimana (Dayname)
        for dayname in df['Dayname'].unique():
            # Filtra il dataframe per il giorno della settimana corrente
            day_df = df[df['Dayname'] == dayname]

            # Raggruppa per "Year", "Month", "Day" e ottieni i profili di carico giornalieri
            daily_profiles = day_df.groupby(["Year", "Month", "Day"])["Norm_consumption"].apply(list).reset_index()

            # Inizializza variabili per l'ottimizzazione del numero di cluster
            best_num_clusters = None
            best_dbi_score = float('inf')

            # Prova il clustering con un numero di cluster variabile da 7 a 20
            for num_clusters in range(7, 21):
                kmeans = KMeans(n_clusters=num_clusters, n_init=100, max_iter=700, algorithm='lloyd')
                clusters = kmeans.fit_predict(list(daily_profiles["Norm_consumption"]))

                # Calcola il Davies-Bouldin Index
                dbi_score = davies_bouldin_score(list(daily_profiles["Norm_consumption"]), clusters)

                # Aggiorna il miglior numero di cluster se necessario
                if dbi_score < best_dbi_score:
                    best_dbi_score = dbi_score
                    best_num_clusters = num_clusters

            # Esegui nuovamente il clustering con il numero ottimale di cluster
            kmeans = KMeans(n_clusters=best_num_clusters, n_init=100, max_iter=700, algorithm='lloyd')
            clusters = kmeans.fit_predict(list(daily_profiles["Norm_consumption"]))

            # Calcola il massimo numero di utenti e profili di carico giornalieri
            max_users = df["User"].nunique()
            max_profiles = len(df["Norm_consumption"])

            # Identifica i cluster poco popolati
            low_populated_clusters = []
            for cluster_id in range(best_num_clusters):
                cluster_users = day_df[clusters == cluster_id]["User"].nunique()
                cluster_profiles = sum(clusters == cluster_id)
                if cluster_users < 0.05 * max_users or cluster_profiles < 0.05 * max_profiles:
                    low_populated_clusters.append(cluster_id)

            # Escludi i profili di carico giornalieri associati ai cluster poco popolati
            updated_day_df = day_df[~clusters.isin(low_populated_clusters)]

            # Aggiorna il dataframe risultante
            result_df = pd.concat([result_df, updated_day_df])

        return result_df

    def merge_clusters(main_df, smaller_df):
        merged_df = pd.merge(main_df, smaller_df[['User', 'Year', 'Month', 'Hour', 'Cluster']],
                             on=['User', 'Year', 'Month', 'Hour'], how='left')
        main_df['Cluster'] = merged_df['Cluster']

        return main_df
