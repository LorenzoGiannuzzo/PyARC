import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt


def kmeans_clustering(df):
    # Creazione dei profili di carico giornalieri
    daily_profiles = df.groupby(["User", "Year", "Month", "Day"]).agg(
        {"Hour": list, "Norm_consumption": list}).reset_index()

    # Inizializzazione dei valori per la ricerca del miglior numero di cluster
    best_davies_bouldin = float('inf')
    best_k = 0

    # Ricerca del numero ottimale di cluster (tra 3 e 7)
    for k in range(3, 8):

            kmeans = KMeans(n_clusters=k, n_init=400, max_iter=700,algorithm='lloyd').fit(daily_profiles["Norm_consumption"].tolist())
            davies_bouldin = davies_bouldin_score(daily_profiles["Norm_consumption"].tolist(), kmeans.labels_)

            if davies_bouldin < best_davies_bouldin:
                best_davies_bouldin = davies_bouldin
                best_k = k

    # Esecuzione del clustering con il miglior numero di cluster
    kmeans = KMeans(n_clusters=best_k, n_init=1, max_iter=700,algorithm='lloyd').fit(daily_profiles["Norm_consumption"].tolist())
    daily_profiles["Cluster"] = kmeans.labels_ +1

    cluster_centroids_df = kmeans.cluster_centers_

    df2 = pd.DataFrame(cluster_centroids_df)
    df2["Cluster"] = df2.index + 1
    df2_long = pd.melt(df2, id_vars=['Cluster'], var_name='Hour', value_name='Norm_consumption')

    # Ordinamento in base a "Cluster" e "Hour"
    df2_long.sort_values(['Cluster', 'Hour'], inplace=True)
    #df_long = pd.melt(df2, var_name="Hour", value_name="Norm_consumption")

    return df.merge(daily_profiles[["User", "Year", "Month", "Day", "Cluster"]],
                    on=["User", "Year", "Month", "Day"]),df2_long

