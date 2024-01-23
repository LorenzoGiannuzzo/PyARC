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

    import pandas as pd
    import numpy as np

    def replace_max_daily_zero_consumption(dataframe):

        # Identify daily profiles with maximum zero value
        max_zero_mask = (dataframe["Consumption"] == 0) & ~dataframe["Consumption"].isna()
        max_zero_profiles = dataframe[max_zero_mask].groupby(["User", "Year", "Month", "Day"])["Consumption"].idxmax()

        # Replace corresponding values with NaN
        dataframe.loc[max_zero_profiles, "Consumption"] = np.nan

        return dataframe



