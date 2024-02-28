import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

class Plots:
    @staticmethod
    def plot_norm_avg_cons(data):
        # Utilizza il backend "agg" per evitare la visualizzazione interattiva
        matplotlib.use('agg')

        plt.figure(figsize=(12, 6))
        sns.lineplot(x="Hour", y="M_consumption", data=data, hue="User", errorbar=None)

        # Adding labels and title
        plt.xlabel("Hour")
        plt.ylabel("Normalized Average Monthly Consumption")
        plt.title("Normalized Average Monthly Consumption Profiles")
        plt.ylim(0, 1)

        # Save the plot as a .png file in the "plots" directory
        script_dir = os.path.dirname(__file__)  # Get the directory of the current script
        plots_dir = os.path.join(script_dir, "..", "plots")  # Navigate to the "plots" directory
        os.makedirs(plots_dir, exist_ok=True)  # Create the "plots" directory if it doesn't exist

        plt.savefig(os.path.join(plots_dir, "Normalized Average Monthly Consumption Profiles.png"))

    @staticmethod
    def plot_cluster_centroids(cluster_centers_long_df):
        # Numero dinamico di colonne in FacetGrid
        num_clusters = cluster_centers_long_df['Cluster'].nunique()

        # Creazione di una palette di colori unica per ciascun cluster
        palette = sns.color_palette("husl", n_colors=num_clusters)

        # Inizializza un FacetGrid
        g = sns.FacetGrid(cluster_centers_long_df, col="Cluster", col_wrap=num_clusters, height=4)

        # Disegna i profili dei centroidi per ogni cluster utilizzando colori diversi
        for i, (_, data) in enumerate(cluster_centers_long_df.groupby('Cluster')):
            sns.lineplot(data=data, x="Hour", y="Centroid", color=palette[i], ax=g.axes[i])

        # Aggiungi etichette e titolo
        g.set_axis_labels("Hour", "Centroid Value")
        g.fig.suptitle("Cluster Centroids Profiles", y=1.1) # Aumentato il valore y per evitare il taglio del titolo

        # Regola gli assi per una migliore visualizzazione
        g.set(xticks=list(range(24)), xlim=(0, 23), ylim=(0, 1))

        # Riduci ulteriormente il margine superiore per evitare il taglio del titolo
        plt.subplots_adjust(top=0.9)

        # Ottieni il percorso della directory del codice
        script_dir = os.path.dirname(__file__)

        # Crea la directory "plots" se non esiste
        plots_dir = os.path.join(script_dir, "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Salva il plot come immagine nella directory "plots"
        plt.savefig(os.path.join(plots_dir, "Cluster_Centroids_Profiles.png"))

    @staticmethod
    def plot_aggregate_loads(dataframe):
        # Assuming 'dataframe' is your pandas DataFrame
        plt.figure(figsize=(12, 8))

        # Iterate over unique months in the DataFrame
        for month in dataframe['Month'].unique():
            plt.subplot(3, 4, month)  # Adjust the subplot grid as needed

            # Filter data for the current month
            month_data = dataframe[dataframe['Month'] == month]

            # Create a box plot for the current month

            # Add a line plot for each day in the month with a more prominent color
            sns.lineplot(x='Hour', y='Aggregate load', data=month_data, legend=None
                         , dashes=False, linewidth=1.5)

            plt.title(f'Month: {month}', fontsize=14)
            plt.xlabel('Hour', fontsize=12)

            # Set x-axis tick labels fontsize and rotation
            plt.xticks(range(24),fontsize=6, rotation=90, ha='right')
            plt.ylabel('Aggregate load [kWh]', fontsize=12)

            # Customize legend (if needed)
            # plt.legend(title='Day', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()

        # Save the plot with a high resolution
        script_dir = os.path.dirname(__file__)
        plots_dir = os.path.join(script_dir, "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "Aggregate_load_profiles.png"), dpi=700)






