# Import necessary libraries
import pandas as pd
import os
from sklearn.metrics import davies_bouldin_score, silhouette_score
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt


# Set the number of cores for joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

class Clustering:

    @staticmethod
    def find_optimal_cluster_number(data):
        try:
            # Check if the required column 'M_consumption' exists in the DataFrame
            if 'M_consumption' not in data.columns:
                raise ValueError("Column 'M_consumption' not found in the DataFrame.")

            # Select only the "M_consumption" column for clustering
            features = data.pivot_table(values='M_consumption', index=['User', 'Year', 'Month'],
                                        columns='Hour').reset_index()

            # Lists to store the results of metrics
            dbi_scores, silhouette_scores, distortion_scores = [], [], []

            # Try clustering for a number of clusters from 4 to 8
            for num_clusters in range(4, 9):
                kmeans = KMeans(n_clusters=num_clusters, n_init=700, max_iter=700, algorithm='lloyd')
                kmeans_labels = kmeans.fit_predict(features.iloc[:, 3:])

                # Calculate and save metric values
                dbi_score = davies_bouldin_score(features.iloc[:, 3:], kmeans_labels)
                silhouette_score_val = silhouette_score(features.iloc[:, 3:], kmeans_labels)
                distortion_score = kmeans.inertia_

                dbi_scores.append(dbi_score)
                silhouette_scores.append(silhouette_score_val)
                distortion_scores.append(distortion_score)

            # Find the optimal number of clusters using the metrics
            optimal_num_clusters_dbi = np.argmin(dbi_scores) + 4
            optimal_num_clusters_silhouette = np.argmax(silhouette_scores) + 4
            optimal_num_clusters_elbow = np.argmin(distortion_scores) + 4

            # Create the plot for the Davies-Bouldin Index trend
            matplotlib.use('agg')
            plt.figure(figsize=(12, 6))
            plt.plot(range(4, 9), dbi_scores, marker='o', linestyle='-', color='b')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Davies-Bouldin Index")
            plt.title("Davies-Bouldin Index vs Number of Clusters")
            plt.grid(True)
            plt.savefig(os.path.join("..", "plots", "Davies-Bouldin Index vs Number of Clusters.png"))

            # Create the plot for the Silhouette Score trend
            plt.figure(figsize=(12, 6))
            plt.plot(range(4, 9), silhouette_scores, marker='o', linestyle='-', color='g')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score vs Number of Clusters")
            plt.grid(True)
            plt.savefig(os.path.join("..", "plots", "Silhouette Score vs Number of Clusters.png"))

            # Create the plot for the Elbow Method (Distortion) trend
            plt.figure(figsize=(12, 6))
            plt.plot(range(4, 9), distortion_scores, marker='o', linestyle='-', color='r')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Distortion (Elbow Method)")
            plt.title("Elbow Method (Distortion) vs Number of Clusters")
            plt.grid(True)
            plt.savefig(os.path.join("..", "plots", "Elbow Method vs Number of Clusters.png"))

            # Count occurrences of each cluster number
            votes = {
                optimal_num_clusters_dbi: 0,
                optimal_num_clusters_silhouette: 0,
                optimal_num_clusters_elbow: 0
            }

            # Increment votes for each cluster number
            for vote in [optimal_num_clusters_dbi, optimal_num_clusters_silhouette, optimal_num_clusters_elbow]:
                votes[vote] += 1

            # Find the optimal number of clusters based on the consensus among metrics
            optimal_num_clusters = max(votes, key=votes.get)

            # In case of a tie, always choose the optimal number of clusters based on DBI
            if list(votes.values()).count(votes[optimal_num_clusters]) > 1:
                optimal_num_clusters = min(votes, key=votes.get)

            return optimal_num_clusters, votes

        except Exception as e:
            return f"Error: {str(e)}", None

    @staticmethod
    def kmeans_clustering(df, optimal_cluster_number):
        try:
            # Check if the required column 'M_consumption' exists in the DataFrame
            if 'M_consumption' not in df.columns:
                raise ValueError("Column 'M_consumption' not found in the DataFrame.")

            # Create a temporary dataframe for clustering features
            features = df.pivot_table(values='M_consumption', index=['User', 'Year', 'Month'], columns='Hour').reset_index()

            # Execute the k-means clustering
            kmeans = KMeans(n_clusters=optimal_cluster_number, n_init=700, max_iter=700, algorithm='lloyd')
            features['Cluster'] = kmeans.fit_predict(features.iloc[:, 3:]) + 1

            # Obtain the centroids of the clusters
            cluster_centers = kmeans.cluster_centers_

            # Add cluster centroids to the result dataframe
            cluster_centers_df = pd.DataFrame(cluster_centers,
                                              columns=[f'Hour_{i}' for i in range(cluster_centers.shape[1])])
            cluster_centers_df.insert(0, 'Cluster', range(1, optimal_cluster_number + 1))

            # Rename the columns of the "Hour" column
            cluster_centers_df.columns = ['Cluster'] + list(range(cluster_centers.shape[1]))

            # Transpose the dataframe into "long" format
            cluster_centers_long_df = pd.melt(cluster_centers_df, id_vars=['Cluster'], var_name='Hour',
                                              value_name='Centroid')

            # Sort the centroids dataframe by "Cluster" and "Hour"
            cluster_centers_long_df = cluster_centers_long_df.sort_values(by=['Cluster', 'Hour']).reset_index(drop=True)

            # Merge the results with the original dataframe
            result_df = pd.merge(df, features[['User', 'Year', 'Month', 'Cluster']], on=['User', 'Year', 'Month'],
                                 how='left')

            return result_df, cluster_centers_long_df

        except Exception as e:
            return f"Error: {str(e)}", None



