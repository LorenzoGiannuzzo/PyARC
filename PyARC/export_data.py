import pandas as pd
import os


class Export:

    def export_centroid_csv(data):
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, "..", "data", "Centroids")
        os.makedirs(data_dir, exist_ok=True)

        # Definisci il percorso del file CSV
        csv_path = os.path.join(data_dir, "centroid_data.csv")

        # Esporta il dataframe in formato CSV
        data.to_csv(csv_path, index=False)

    def export_output_csv(data):
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, "..", "data", "Output Data")
        os.makedirs(data_dir, exist_ok=True)

        # Definisci il percorso del file CSV
        csv_path = os.path.join(data_dir, "Aggregated Data.csv")

        # Esporta il dataframe in formato CSV
        data.to_csv(csv_path, index=False)