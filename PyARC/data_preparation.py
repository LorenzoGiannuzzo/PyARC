# Import necessary libraries
import pandas as pd


class DataFrameProcessor:
    def __init__(self, dataframe):
        # Constructor to initialize the DataFrameProcessor object with a DataFrame
        self.dataframe = dataframe

    def process_dataframe(self):
        """
        Process the DataFrame by performing the following steps:
        1. Check if required columns are present.
        2. Create a new column "Dayname" based on "Year", "Month", and "Day".
        3. Drop the intermediate "Date" column if not needed.

        Returns:
        - Processed DataFrame.
        """
        # Check if required columns are present
        required_columns = ["User", "Year", "Month", "Day", "Hour", "Consumption"]
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]

        if missing_columns:
            raise ValueError(f"Error: Missing columns {missing_columns}")

        # Create a new column "Dayname" based on "Year", "Month", and "Day"
        self.dataframe['Date'] = pd.to_datetime(self.dataframe[['Year', 'Month', 'Day']])
        self.dataframe['Dayname'] = self.dataframe['Date'].dt.day_name()

        # Drop the intermediate "Date" column if not needed
        self.dataframe = self.dataframe.drop(columns=['Date'])

        return self.dataframe

    @staticmethod
    def data_subset(data):
        """
        Create subsets of data based on the value of the "Month" column.

        Args:
        - data: Input DataFrame.

        Returns:
        - Data subsets for summer, winter, spring, and autumn.
        """
        # Create subsets based on the value of the "Month" column
        data_summer = data[data['Month'].isin([6, 7, 8])]
        data_winter = data[data['Month'].isin([1, 2, 12])]
        data_spring = data[data['Month'].isin([3, 4, 5])]
        data_autumn = data[data['Month'].isin([9, 10, 11])]

        # Return the created subsets
        return data_summer, data_winter, data_spring, data_autumn

    @staticmethod
    def check_tou(dataframe):
        """
        Check if the "Hour" and "ToU" columns are present in the DataFrame.

        Args:
        - dataframe: Input DataFrame.

        Raises:
        - ValueError: If "Hour" or "ToU" columns are not present in the DataFrame.
        """
        # Check if the "Hour" and "ToU" columns are present in the DataFrame
        required_columns = ["Hour", "ToU"]

        if all(col in dataframe.columns for col in required_columns):
            print("Columns 'Hour' and 'ToU' are present.")
        else:
            raise ValueError("The 'Hour' or 'ToU' columns are not present in the DataFrame.")
