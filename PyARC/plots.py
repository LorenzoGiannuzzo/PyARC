import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors


def create_custom_palette(num_clusters):
    # Definisci una palette di colori personalizzata
    custom_palette = sns.color_palette("viridis", num_clusters)  # Puoi scegliere diversi schemi di colori

    # Puoi regolare la luminosità e la saturazione dei colori
    custom_palette = mcolors.rgb_to_hsv(custom_palette)
    custom_palette[:, 1] = 0.8  # Regola la saturazione
    custom_palette[:, 2] = 0.9  # Regola la luminosità
    custom_palette = mcolors.hsv_to_rgb(custom_palette)

    return custom_palette

class Plots:
    @staticmethod
    def plot_norm_avg_cons(data):
        # Utilizza il backend "agg" per evitare la visualizzazione interattiva
        matplotlib.use('agg')

        plt.figure(figsize=(12, 8))
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
        # Dynamic number of columns in FacetGrid
        num_clusters = cluster_centers_long_df['Cluster'].nunique()

        plt.figure(figsize=(12, 8))

        # Crea una palette di colori personalizzata
        custom_palette = create_custom_palette(num_clusters)

        # Initialize a FacetGrid
        g = sns.FacetGrid(cluster_centers_long_df, col="Cluster", hue="Cluster", col_wrap=num_clusters, height=4,
                          palette=custom_palette)

        # Draw centroid profiles for each cluster using different colors
        g.map(sns.lineplot, "Hour", "Centroid")

        # Add labels and title
        g.set_axis_labels("Hour", "Normalized Consumption")
        g.fig.suptitle("Cluster Centroids Profiles", y=1.1)  # Increased y-value to avoid title cutoff

        # Adjust axes for better visualization
        g.set(xticks=list(range(24)), xlim=(0, 23), ylim=(0, 1))

        # Further reduce the top margin to avoid title cutoff
        plt.subplots_adjust(top=0.9)

        # Rotate x-axis labels on all subplots
        for ax in g.axes.flatten():
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=90, ha='right')

        # Get the directory path of the code
        script_dir = os.path.dirname(__file__)

        # Create the "plots" directory if it doesn't exist
        plots_dir = os.path.join(script_dir, "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()

        # Save the plot as an image in the "plots" directory
        plt.savefig(os.path.join(plots_dir, "Cluster_Centroids_Profiles.png"), dpi=700)

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
                         , dashes=False, linewidth=1.5, color= 'green')

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






