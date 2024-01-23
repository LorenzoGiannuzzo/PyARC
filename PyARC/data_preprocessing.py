import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_negative_values(self):
        # Check if the "Consumption" column exists in the dataframe
        if "Consumption" in self.dataframe.columns:
            # Replace negative values with NaN in the "Consumption" column
            self.dataframe["Consumption"] = np.where(self.dataframe["Consumption"] < 0, np.nan, self.dataframe["Consumption"])
            return self.dataframe
        else:
            print("The 'Consumption' column is not present in the dataframe.")
            return None
