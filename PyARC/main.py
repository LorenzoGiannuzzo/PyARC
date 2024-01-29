# Import necessary modules
import os
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from get_tou import CSVHandler as TouCSVHandler
from get_file import CSVHandler as DataCSVHandler
from data_preparation import DataFrameProcessor
from data_preprocessing import DataPreprocessing
from data_normalization import DataNormalization
from data_clustering import kmeans_clustering

# Define PyARC class
class PyARC:
    def train_model(self, data_path, tou_path):
        # Load data CSV
        data_handler = DataCSVHandler(data_path)
        data_handler.load_csv()
        data_dataframe = data_handler.get_data()

        # Load tou CSV
        tou_handler = TouCSVHandler(tou_path)
        tou_handler.load_csv()
        tou_dataframe = tou_handler.get_data()

        data = DataFrameProcessor(data_dataframe)
        data = data.process_dataframe()

        DataFrameProcessor.check_tou(tou_dataframe)

        # Creating a DataPreprocessing object
        data_processor = DataPreprocessing(data)

        # Calling the get_negative_values function
        corrected_data = data_processor.get_negative_values()
        corrected_data = DataPreprocessing.replace_max_daily_zero_consumption(corrected_data)

        corrected_data = DataPreprocessing.interpolate_missing_values(corrected_data,max_gap=3)
        corrected_data = DataPreprocessing.fill_missing_values_with_monthly_mean(corrected_data)

        corrected_data = DataPreprocessing.remove_outliers_iqr(corrected_data)

        corrected_data = DataPreprocessing.interpolate_missing_values(corrected_data, max_gap=3)
        corrected_data = DataPreprocessing.fill_missing_values_with_monthly_mean(corrected_data)


        data_normalizer = DataNormalization(corrected_data)
        corrected_data = data_normalizer.normalize_consumption()

        corrected_data = DataPreprocessing.filter_users(corrected_data)

        data_summer, data_winter, data_spring, data_autumn = DataFrameProcessor.data_subset(corrected_data)

       # data_summer2 = DataPreprocessing.infrequent_profiles(data_summer)  -> lo tengo in stand_by per adesso

        result_df, centroids = kmeans_clustering(corrected_data)

        result_df['Hour'] = pd.to_numeric(result_df['Hour'])

        # Creazione della griglia di lineplot
        g = sns.FacetGrid(result_df, col="Cluster", col_wrap=3, height=4, sharey=False, margin_titles=True)

        # Map su tutti i profili di carico giornalieri, impostando il colore a grigio
        g.map_dataframe(sns.lineplot, x="Hour", y="Norm_consumption", hue="User", alpha=0.3)

        # Impostazioni estetiche
        g.set_axis_labels("Hour", "Norm_consumption")
        g.set_titles(col_template="Cluster {col_name}")
        g.fig.suptitle("Lineplot for each Cluster", y=1.02)


        # Impostazione della scala uniforme sulle y
        g.set(ylim=(0, 1))

        plt.show()

        # cluster_colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'yellow', 5: 'grey'}  # Puoi estendere questo dizionario in base ai tuoi dati effettivi
        #
        # # Creazione di una griglia di line plot per ogni cluster
        # g = sns.FacetGrid(result_df, col="Cluster", col_wrap=3, height=4, sharey=False, margin_titles=True,
        #                   palette=cluster_colors)
        # g.map(sns.lineplot, "Hour", "Norm_consumption", sort=False)
        #
        # # Impostazioni estetiche
        # g.set_axis_labels("Hour", "Norm_consumption")
        # g.set_titles(col_template="{col_name}")
        # g.fig.suptitle("Line Plot per Cluster", y=1.02)
        #
        # # Impostazione della scala uniforme sulle y
        # g.set(ylim=(0, 1))
        #
        # plt.show()

        return result_df

# Check if the script is being run as the main program
if __name__ == "__main__":
    # Calculate full paths to the files
    data_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    data_file_path = os.path.join(data_file_directory, "default_data.csv")

    tou_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    tou_file_path = os.path.join(tou_file_directory, "default_tou.csv")

    # Create an instance of PyARC and call train_model
    pyarc_instance = PyARC()
    corrected_data = pyarc_instance.train_model(data_file_path, tou_file_path)

