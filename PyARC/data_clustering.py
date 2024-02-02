import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from kneed import KneeLocator
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
import matplotlib

import matplotlib.pyplot as plt

class Clustering:

    def find_optimal_cluster_number(data):
        # Seleziona solo la colonna "M_consumption" per il clustering
        features = data.pivot_table(values='M_consumption', index=['User', 'Year', 'Month'],
                                    columns='Hour').reset_index()

        # Lista per memorizzare i risultati delle metriche
        dbi_scores, silhouette_scores, distortion_scores = [], [], []

        # Prova il clustering per un numero di cluster da 4 a 8
        for num_clusters in range(4, 9):
            kmeans = KMeans(n_clusters=num_clusters, n_init=700, max_iter = 700, algorithm='lloyd')
            kmeans_labels = kmeans.fit_predict(features.iloc[:, 3:])

            # Calcola e salva i valori delle metriche
            dbi_score = davies_bouldin_score(features.iloc[:, 3:], kmeans_labels)
            silhouette_score_val = silhouette_score(features.iloc[:, 3:], kmeans_labels)
            distortion_score = kmeans.inertia_

            dbi_scores.append(dbi_score)
            silhouette_scores.append(silhouette_score_val)
            distortion_scores.append(distortion_score)

        # Trova il numero ottimale di cluster utilizzando le metriche
        optimal_num_clusters_dbi = np.argmin(dbi_scores) + 4
        optimal_num_clusters_silhouette = np.argmax(silhouette_scores) + 4
        optimal_num_clusters_elbow = np.argmin(distortion_scores) + 4

        # Crea il plot per l'andamento del Davies-Bouldin Index
        matplotlib.use('agg')
        plt.figure(figsize=(12, 6))
        plt.plot(range(4, 9), dbi_scores, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Davies-Bouldin Index")
        plt.title("Davies-Bouldin Index vs Number of Clusters")
        plt.grid(True)
        plt.savefig(os.path.join("..", "plots", "Davies-Bouldin Index vs Number of Clusters.png"))

        # Crea il plot per l'andamento del Silhouette Score
        plt.figure(figsize=(12, 6))
        plt.plot(range(4, 9), silhouette_scores, marker='o', linestyle='-', color='g')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs Number of Clusters")
        plt.grid(True)
        plt.savefig(os.path.join("..", "plots", "Silhouette Score vs Number of Clusters.png"))


        # Crea il plot per l'andamento dell'Elbow Method (Distortion)
        plt.figure(figsize=(12, 6))
        plt.plot(range(4, 9), distortion_scores, marker='o', linestyle='-', color='r')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Distortion (Elbow Method)")
        plt.title("Elbow Method (Distortion) vs Number of Clusters")
        plt.grid(True)
        plt.savefig(os.path.join("..", "plots", "Elbow Method vs Number of Clusters.png"))

        # Conta le occorrenze di ciascun numero di cluster
        votes = {
            optimal_num_clusters_dbi: 0,
            optimal_num_clusters_silhouette: 0,
            optimal_num_clusters_elbow: 0
        }

        # Incrementa i voti per ciascun numero di cluster
        for vote in [optimal_num_clusters_dbi, optimal_num_clusters_silhouette, optimal_num_clusters_elbow]:
            votes[vote] += 1

        # Trova il numero ottimale di cluster basato sul consenso tra le metriche
        optimal_num_clusters = max(votes, key=votes.get)

        # In caso di parità, scegli sempre il numero ottimale di cluster basato sul DBI
        if list(votes.values()).count(votes[optimal_num_clusters]) > 1:
            optimal_num_clusters = min(votes, key=votes.get)

        return optimal_num_clusters, votes

    def kmeans_clustering(df, optimal_cluster_number):
        # Creazione di un dataframe temporaneo per le features di clustering
        features = df.pivot_table(values='M_consumption', index=['User', 'Year', 'Month'], columns='Hour').reset_index()

        # Esecuzione del k-means clustering
        kmeans = KMeans(n_clusters=optimal_cluster_number, n_init=700, max_iter=700, algorithm='lloyd')
        features['Cluster'] = kmeans.fit_predict(features.iloc[:, 3:]) + 1

        # Otteniamo i centroidi dei cluster
        cluster_centers = kmeans.cluster_centers_

        # Aggiungiamo i centroidi dei cluster al dataframe risultato
        cluster_centers_df = pd.DataFrame(cluster_centers,
                                          columns=[f'Hour_{i}' for i in range(cluster_centers.shape[1])])
        cluster_centers_df.insert(0, 'Cluster', range(1, optimal_cluster_number + 1))

        # Ridenomina le colonne della colonna "Hour"
        cluster_centers_df.columns = ['Cluster'] + list(range(cluster_centers.shape[1]))

        # Trasponi il dataframe in formato "long"
        cluster_centers_long_df = pd.melt(cluster_centers_df, id_vars=['Cluster'], var_name='Hour',
                                          value_name='Centroid')

        # Ordina il dataframe dei centroidi per "Cluster" e "Hour"
        cluster_centers_long_df = cluster_centers_long_df.sort_values(by=['Cluster', 'Hour']).reset_index(drop=True)

        # Uniamo i risultati al dataframe originale
        result_df = pd.merge(df, features[['User', 'Year', 'Month', 'Cluster']], on=['User', 'Year', 'Month'],
                             how='left')

        return result_df, cluster_centers_long_df

