import pandas as pd
from datetime import datetime

class DataFrameProcessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def process_dataframe(self):
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
