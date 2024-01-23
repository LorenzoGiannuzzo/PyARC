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

    def data_subset(data):
        # Creazione dei subset in base al valore della colonna "Month"
        data_summer = data[data['Month'].isin([6, 7, 8])]
        data_winter = data[data['Month'].isin([1, 2, 12])]
        data_spring = data[data['Month'].isin([3, 4, 5])]
        data_autumn = data[data['Month'].isin([9, 10, 11])]

        # Ritorna i subset creati
        return data_summer, data_winter, data_spring, data_autumn

    def check_tou(dataframe):
        # Check if the "Hour" and "ToU" columns are present in the dataframe
        required_columns = ["Hour", "ToU"]

        if all(col in dataframe.columns for col in required_columns):
            print("")
        else:
            print("The 'Hour' or 'Tou' columns are not present in tou_dataframe")