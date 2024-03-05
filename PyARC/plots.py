import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np


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
        # Dynamic number of columns and rows in FacetGrid
        #
        num_cols = 2  # Set the number of columns per row

        # Reduce the overall size of the image
        plt.figure(figsize=(12, 6))  # Adjust dimensions according to your preferences

        # Create a custom color palette
        custom_palette = sns.color_palette("viridis", n_colors=len(cluster_centers_long_df['Cluster'].unique()))

        # Initialize a FacetGrid with specific columns per row and use the custom palette
        g = sns.FacetGrid(cluster_centers_long_df, col="Cluster", hue="Cluster", col_wrap=num_cols, height=2,
                          palette=custom_palette)

        # Draw centroid profiles for each cluster using different colors
        # Increase the thickness of lineplots
        g.map(sns.lineplot, "Hour", "Centroid", linewidth=2)

        # Add labels and title
        g.set_axis_labels("Hour", "Normalized Consumption")

        # Set ticks to display numbers on the X-axis every 6 hours
        g.set(xticks=list(range(0, 24, 6)), xlim=(0, 23), ylim=(0, 1))

        # Set the grid on all Y-axes
        # for ax in g.axes.flat:
        #     ax.set_yticks(np.arange(0, 1.1, 0.1))

        # Rotate X-axis labels on all subplots
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelrotation=0, labelsize=7)

        # Further reduce the top margin to avoid title cutoff
        plt.subplots_adjust(top=0.9)

        # Get the directory path of the code
        script_dir = os.path.dirname(__file__)

        # Create the "plots" directory if it doesn't exist
        plots_dir = os.path.join(script_dir, "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plt.tight_layout()

        # Save the plot as an image in the "plots" directory
        plt.savefig(os.path.join(plots_dir, "Cluster_Centroids_Profiles.png"), dpi=300)  # Reduce dpi if necessary


    @staticmethod
    def plot_aggregate_loads(dataframe):
        plt.figure(figsize=(12, 8))

        # Define a vibrant color palette with shades of green, blue, and orange
        color_palette = sns.color_palette("viridis", n_colors=len(dataframe['Month'].unique()))

        # Iterate over unique months in the DataFrame
        for idx, month in enumerate(dataframe['Month'].unique()):
            plt.subplot(3, 4, idx + 1)  # Adjust the subplot grid as needed

            # Filter data for the current month
            month_data = dataframe[dataframe['Month'] == month]

            # Create a line plot for the current month with a different color
            sns.lineplot(x='Hour', y='Aggregate load', data=month_data, legend=None,
                         dashes=False, linewidth=2.5, color=color_palette[idx])

            plt.title(f'Month: {month}', fontsize=14)
            plt.xlabel('Hour', fontsize=12)

            # Set x-axis tick labels fontsize and rotation, display every 6th hour
            plt.xticks(range(0, 24, 6), fontsize=10, rotation=0, ha='right')

            plt.ylabel('Aggregate load [kWh]', fontsize=12)

            # Remove the grid from the plot
            plt.grid(False)

            plt.tight_layout()

        # Save the plot with a high resolution
        script_dir = os.path.dirname(__file__)
        plots_dir = os.path.join(script_dir, "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "Aggregate_load_profiles.png"), dpi=700)







