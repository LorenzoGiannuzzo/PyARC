# Import necessary libraries
import numpy as np
import pandas as pd

class DataPreprocessing:
    def __init__(self, dataframe):
        # Initialize the DataPreprocessing class with the input DataFrame
        self.dataframe = dataframe

    def get_negative_values(self):
        # Replace negative values in the 'Consumption' column with NaN
        if "Consumption" in self.dataframe.columns:
            self.dataframe["Consumption"] = np.where(self.dataframe["Consumption"] < 0, np.nan, self.dataframe["Consumption"])
            return self.dataframe
        else:
            print("The 'Consumption' column is not present in the dataframe.")
            return None

    @staticmethod
    def replace_max_daily_zero_consumption(dataframe):
        # Replace the maximum daily zero consumption with NaN
        max_zero_mask = (dataframe["Consumption"] == 0) & ~dataframe["Consumption"].isna()
        max_zero_profiles = dataframe[max_zero_mask].groupby(["User", "Year", "Month", "Day"])["Consumption"].idxmax()
        dataframe.loc[max_zero_profiles, "Consumption"] = np.nan
        return dataframe

    @staticmethod
    def interpolate_missing_values(df, max_gap=3):
        # Interpolate missing values in the 'Consumption' column within user groups
        df = df.sort_values(by=["User", "Year", "Month", "Day"])
        df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce')

        grouped_df = df.groupby("User")

        def interpolate_group(group):
            group['Consumption'] = group['Consumption'].interpolate(method='linear', limit=max_gap)
            return group

        interpolated_df = grouped_df.apply(interpolate_group)
        interpolated_df.reset_index(drop=True, inplace=True)

        return interpolated_df

    @staticmethod
    def fill_missing_values_with_monthly_mean(df):
        # Fill missing values in the 'Consumption' column with monthly means
        df = df.sort_values(by=["User", "Year", "Month", "Day"])
        monthly_means = df.groupby(["User", "Year", "Month"])["Consumption"].mean().reset_index()
        df_filled = pd.merge(df, monthly_means, on=["User", "Year", "Month"], how="left", suffixes=('', '_mean'))
        df_filled["Consumption"] = df_filled["Consumption"].fillna(df_filled["Consumption_mean"])
        df_filled = df_filled.drop(columns=["Consumption_mean"])
        return df_filled

    @staticmethod
    def remove_outliers_iqr(df):
        # Remove outliers in the 'Consumption' column using the Interquartile Range (IQR) method
        column_name = "Consumption"
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[column_name] = df[column_name].apply(lambda x: x if lower_bound <= x <= upper_bound else None)
        return df

    @staticmethod
    def filter_users(df):
        # Filter out users with incomplete hourly data
        df = df.sort_values(by=["User", "Year", "Month", "Day"])
        # Convert the "Hour" column to numeric if not already
        if not pd.api.types.is_numeric_dtype(df['Hour']):
            df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')

        # Group by "User", "Year", "Month", "Day" and check if all hours from 0 to 23 are present
        valid_users_series = df.groupby(["User", "Year", "Month", "Day"])["Hour"].apply(
            lambda x: set(x) == set(range(24)))

        # Create a new DataFrame with columns "User", "Year", "Month", "Day", "Valid"
        valid_df = pd.DataFrame(valid_users_series).reset_index()
        valid_df.columns = ["User", "Year", "Month", "Day", "Valid"]

        # Filter the DataFrame based on valid users
        filtered_df = df[df["User"].isin(valid_df[valid_df["Valid"]]["User"])]

        # Calculate the number of eliminated users
        num_users_eliminated = len(df["User"].unique()) - len(filtered_df["User"].unique())

        # Output message
        print(f"{num_users_eliminated} users eliminated. DataFrame processed successfully.")

        return filtered_df

    @staticmethod
    def monthly_average_consumption(dataframe):
        # Create a copy of the input DataFrame
        result_df = dataframe.copy()

        # Group the DataFrame by User, Year, Month, and Hour
        grouped_df = result_df.groupby(["User", "Year", "Month", "Hour"])

        # Calculate the monthly average normalized consumption
        result_df["M_consumption"] = grouped_df["Norm_consumption"].transform("mean")

        return result_df

    @staticmethod
    def reshape_dataframe(input_df):
        # Group the input DataFrame by 'User', 'Year', 'Month', and 'Hour' and calculate the mean
        grouped_df = input_df.groupby(['User', 'Year', 'Month', 'Hour'])['M_consumption'].mean().reset_index()

        return grouped_df

    @staticmethod
    def merge_clusters(main_df, smaller_df):
        # Merge clusters based on 'User', 'Year', 'Month', and 'Hour'
        merged_df = pd.merge(main_df, smaller_df[['User', 'Year', 'Month', 'Hour', 'Cluster']],
                             on=['User', 'Year', 'Month', 'Hour'], how='left')
        main_df['Cluster'] = merged_df['Cluster']

        return main_df

