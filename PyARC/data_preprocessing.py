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

    import pandas as pd
    import numpy as np

    import pandas as pd
    import numpy as np

    import pandas as pd
    import numpy as np

    def replace_punctual_outliers(data, max_gap=3):
        # Check if the required columns exist in the dataframe
        required_columns = ["Year", "User", "Month", "Day", "Consumption"]
        if not all(col in data.columns for col in required_columns):
            print("Required columns are not present in the dataframe.")
            return None

        # Ensure there are no missing values in the "Consumption" column
        if data["Consumption"].isnull().any():
            print("The dataframe contains missing values in the 'Consumption' column.")
            return None

        # Calculate quantile values (0.25, 0.75) for each group
        quantile_values = data.groupby(["User", "Year", "Month"])["Consumption"].transform(
            lambda x: np.percentile(x, [25, 75]))

        # Calculate IQR (Interquartile Range) values
        iqr_values = abs(quantile_values.groupby(data["User"]).diff())

        # Calculate lower and upper bounds to identify punctual outliers
        lower_bound = quantile_values - 1.5 * iqr_values
        upper_bound = quantile_values + 1.5 * iqr_values

        # Identify and replace punctual outliers with NaN
        outlier_mask = (data["Consumption"] < lower_bound) | (data["Consumption"] > upper_bound)
        data.loc[outlier_mask, "Consumption"] = np.nan

        # Interpolate NaN values using linear interpolation with a maximum gap of max_gap
        data["Consumption"] = data.groupby(["User", "Year", "Month"])["Consumption"].apply(
            lambda x: x.interpolate(method="linear", limit=max_gap))

        return data

    # Example of usage





