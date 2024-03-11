# Import necessary libraries
import os

class Export:
    @staticmethod
    def export_centroid_csv(data):
        # Get the script directory
        script_dir = os.path.dirname(__file__)

        # Define the data directory for centroids
        data_dir = os.path.join(script_dir, "..", "data", "Centroids")
        os.makedirs(data_dir, exist_ok=True)

        # Define the CSV file path
        csv_path = os.path.join(data_dir, "centroid_data.csv")

        # Export the dataframe to a CSV file
        data.to_csv(csv_path, index=False)

    @staticmethod
    def export_output_csv(data):
        # Get the script directory
        script_dir = os.path.dirname(__file__)

        # Define the data directory for output data
        data_dir = os.path.join(script_dir, "..", "data", "Output Data")
        os.makedirs(data_dir, exist_ok=True)

        # Define the CSV file path
        csv_path = os.path.join(data_dir, "Aggregated Data.csv")

        # Export the dataframe to a CSV file
        data.to_csv(csv_path, index=False)
