import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from kneed import KneeLocator
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

class Clustering:

    def find_optimal_cluster_number(data):
        # Seleziona solo la colonna "M_consumption" per il clustering
        X = data[['M_consumption']]

        # Lista per memorizzare i risultati dell'indice di Davies-Bouldin
        dbi_scores = []

        # Prova il clustering per un numero di cluster da 3 a 8
        for num_clusters in range(3, 9):
            kmeans = KMeans(n_clusters=num_clusters, n_init=400, max_iter=600, algorithm='lloyd')
            kmeans_labels = kmeans.fit_predict(X)

            # Calcola e salva il valore dell'indice di Davies-Bouldin
            dbi_score = davies_bouldin_score(X, kmeans_labels)
            dbi_scores.append(dbi_score)

        # Trova il numero ottimale di cluster utilizzando l'indice di Davies-Bouldin
        optimal_num_clusters = np.argmin(
            dbi_scores) + 3  # Argmin restituisce l'indice del minimo, aggiungiamo 3 per ottenere il numero di cluster

        # Crea il plot per l'andamento del DBI index al variare del numero di cluster
        plt.figure(figsize=(12, 6))
        plt.plot(range(3, 9), dbi_scores, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Davies-Bouldin Index")
        plt.title("Davies-Bouldin Index vs Number of Clusters")
        plt.grid(True)

        # Save the plot as a .png file in the "plots" directory
        script_dir = os.path.dirname(__file__)  # Get the directory of the current script
        plots_dir = os.path.join(script_dir, "..", "plots")  # Navigate to the "plots" directory
        os.makedirs(plots_dir, exist_ok=True)  # Create the "plots" directory if it doesn't exist

        plt.savefig(os.path.join(plots_dir, "Davies-Bouldin Index vs Number of Clusters.png"))

        return optimal_num_clusters

    def k_means_clustering(data,n_optimal_cluster):
        # Seleziona solo la colonna "M_consumption" per il clustering
        X = data[['M_consumption']]
        # Prova il clustering per un numero di cluster da 3 a 8

        kmeans = KMeans(n_clusters=n_optimal_cluster, n_init=400, max_iter=600, algorithm='lloyd')
        kmeans_labels = kmeans.fit_predict(X)

        return kmeans_labels

