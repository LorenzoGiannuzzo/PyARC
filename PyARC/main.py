# Import necessary modules
import os
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import pandas as pd
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

        merged_df = pd.merge(result_df, centroids, on='Cluster', how='left', suffixes=('_original', '_centroid'))

        # Plot dei cluster e dei centroidi
        for cluster in merged_df['Cluster'].unique():
            cluster_data = merged_df[merged_df['Cluster'] == cluster]

            plt.figure(figsize=(8, 5))

            # Plot dei consumi originali
            plt.scatter(cluster_data['Hour_original'], cluster_data['Norm_consumption_original'], label='Original Data',
                        color='blue')

            # Plot dei centroidi
            centroid_data = centroids[centroids['Cluster'] == cluster]
            plt.scatter(centroid_data['Hour'], centroid_data['Norm_consumption'], label='Centroid', color='red',
                        marker='x', s=100)

            plt.title(f'Cluster {cluster} - Original vs Centroid')
            plt.xlabel('Hour')
            plt.ylabel('Norm_consumption')
            plt.legend()
            plt.show()

        return result_df, centroids

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

