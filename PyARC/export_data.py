import pandas as pd
import os
import numpy as np
import matplotlib

class Export:

    def export_csv(data):
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, "..", "data", "Centroids")
        os.makedirs(data_dir, exist_ok=True)

        # Definisci il percorso del file CSV
        csv_path = os.path.join(data_dir, "centroid_data.csv")

        # Esporta il dataframe in formato CSV
        data.to_csv(csv_path, index=False)