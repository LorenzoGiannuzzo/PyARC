import os
import pandas as pd
import numpy as np
import joblib

from get_tou import CSVHandler as TouCSVHandler
from get_file import CSVHandler as DataCSVHandler
from data_preparation import DataFrameProcessor
from data_preprocessing import DataPreprocessing
from data_normalization import DataNormalization
from data_clustering import Clustering
from plots import Plots
from export_data import Export
from get_features import GetFeatures
from data_classification import RandomForest
from data_aggregation import Aggregator




class PyARC:
    def train_model(self, data_path, tou_path):
        # Load data CSV
        data_handler = DataCSVHandler(data_path)
        data_handler.load_csv()
        data_dataframe = data_handler.get_data()

        # Load TOU (Time of Use) CSV
        tou_handler = TouCSVHandler(tou_path)
        tou_handler.load_csv()
        tou_dataframe = tou_handler.get_data()

        # Process data using DataFrameProcessor
        data = DataFrameProcessor(data_dataframe)
        data = data.process_dataframe()

        # Check TOU data
        DataFrameProcessor.check_tou(tou_dataframe)

        # Data preprocessing steps
        data_processor = DataPreprocessing(data)
        data = data_processor.get_negative_values()
        data = DataPreprocessing.replace_max_daily_zero_consumption(data)
        data = DataPreprocessing.interpolate_missing_values(data, max_gap=3)
        data = DataPreprocessing.fill_missing_values_with_monthly_mean(data)
        data = DataPreprocessing.remove_outliers_iqr(data)
        data = DataPreprocessing.interpolate_missing_values(data, max_gap=3)
        data = DataPreprocessing.fill_missing_values_with_monthly_mean(data)

        # Normalize consumption data
        data_normalizer = DataNormalization(data)
        data = data_normalizer.normalize_consumption()

        # Filter users
        data = DataPreprocessing.filter_users(data)

        # Monthly average consumption and reshape dataframe
        data = DataPreprocessing.monthly_average_consumption(data)
        data_monthly = DataPreprocessing.reshape_dataframe(data)

        # Plot normalized average consumption
        Plots.plot_norm_avg_cons(data_monthly)

        # Find optimal number of clusters and perform K-means clustering
        optimal_number_cluster, votes = Clustering.find_optimal_cluster_number(data_monthly)
        data_monthly, centroids = Clustering.kmeans_clustering(data_monthly, optimal_number_cluster)

        # Merge clusters and plot cluster centroids
        data = DataPreprocessing.merge_clusters(data, data_monthly)
        Plots.plot_cluster_centroids(centroids)

        # Export cluster centroids to CSV
        Export.export_centroid_csv(centroids)

        # Spot TOU values in the corrected data
        data = GetFeatures.spot_tou(data, tou_dataframe)

        # Extract features, create permutation ratios, and select features
        data = GetFeatures.get_features(data)
        data = GetFeatures.create_permutation_ratios(data)

        features = GetFeatures.get_selected_features_and_cluster(data)

        # Train a Random Forest model using selected features
        model = RandomForest.model_training(features)

        # Return the trained model
        return model

    def reconstruct_profiles(self):
        model_path = os.path.join("..", "Pre-trained Model", "random_forest_model.joblib")

        # Load the saved model
        model = joblib.load(model_path)

        data_path = os.path.join("..", "data", "Input Data", "data.csv")
        tou_path = os.path.join("..", "data", "Input Data", "tou.csv")
        centroids_path = os.path.join("..","data","Centroids","centroid_data.csv")

        data_handler = DataCSVHandler(data_path)
        data_handler.load_csv()
        data = data_handler.get_data()

        # Load TOU (Time of Use) CSV
        tou_handler = TouCSVHandler(tou_path)
        tou_handler.load_csv()
        tou = tou_handler.get_data()

        centroids_handler = DataCSVHandler(centroids_path)
        centroids_handler.load_csv()
        centroids = centroids_handler.get_data()

        # Extract features, create permutation ratios, and select features

        data = GetFeatures.get_features2(data)

        # Get feature names used by the model
        feature_names_used = model.feature_names_in_
        # Keep only the columns recognized by the model

        features = data[feature_names_used]

        features = features.loc[(features != 0).any(axis=1)]
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.fillna(0, inplace=True)
        features = features.drop_duplicates()

        data = data.loc[(data != 0).any(axis=1)]
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.fillna(0, inplace=True)

        features["Cluster"] = model.predict(features)

        merge_columns = [col for col in features.columns if col != 'Cluster']

        # Effettua la fusione basata sulle colonne chiave
        merged_data = data.merge(features[['Cluster'] + merge_columns], how='left', on=merge_columns)

        centroids = GetFeatures.spot_tou(centroids, tou)
        centroids = GetFeatures.identify_main_ToU(centroids)
        centroids = GetFeatures.calculate_sum_column(centroids)
        centroids = GetFeatures.calculate_weight_coefficient(centroids)
        centroids = GetFeatures.numeric_to_words(centroids)

        output = Aggregator.expand_dataframe(merged_data)
        output = pd.merge(output, centroids, on='Cluster', how='inner')
        output = Aggregator.load_profile_generator(output)
        output = Aggregator.aggregate_load(output)

        Plots.plot_aggregate_loads(output)

        Export.export_output_csv(output)

        return output

    def use_user_trained_model(self):
        model_path = os.path.join("..", "User-trained Model", "random_forest_model.joblib")

        # Load the saved model
        model = joblib.load(model_path)

        data_path = os.path.join("..", "data", "Input Data", "data.csv")
        tou_path = os.path.join("..", "data", "Input Data", "tou.csv")
        centroids_path = os.path.join("..", "data", "Centroids", "centroid_data.csv")

        data_handler = DataCSVHandler(data_path)
        data_handler.load_csv()
        data = data_handler.get_data()

        # Load TOU (Time of Use) CSV
        tou_handler = TouCSVHandler(tou_path)
        tou_handler.load_csv()
        tou = tou_handler.get_data()

        centroids_handler = DataCSVHandler(centroids_path)
        centroids_handler.load_csv()
        centroids = centroids_handler.get_data()

        # Extract features, create permutation ratios, and select features

        data = GetFeatures.get_features2(data)

        # Get feature names used by the model
        feature_names_used = model.feature_names_in_
        # Keep only the columns recognized by the model

        features = data[feature_names_used]

        features = features.loc[(features != 0).any(axis=1)]
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.fillna(0, inplace=True)
        features = features.drop_duplicates()

        data = data.loc[(data != 0).any(axis=1)]
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.fillna(0, inplace=True)

        features["Cluster"] = model.predict(features)

        merge_columns = [col for col in features.columns if col != 'Cluster']

        # Effettua la fusione basata sulle colonne chiave
        merged_data = data.merge(features[['Cluster'] + merge_columns], how='left', on=merge_columns)

        centroids = GetFeatures.spot_tou(centroids, tou)
        centroids = GetFeatures.identify_main_ToU(centroids)
        centroids = GetFeatures.calculate_sum_column(centroids)
        centroids = GetFeatures.calculate_weight_coefficient(centroids)
        centroids = GetFeatures.numeric_to_words(centroids)

        output = Aggregator.expand_dataframe(merged_data)
        output = pd.merge(output, centroids, on='Cluster', how='inner')
        output = Aggregator.load_profile_generator(output)
        output = Aggregator.aggregate_load(output)

        Plots.plot_aggregate_loads(output)

        Export.export_output_csv(output)

        return output
