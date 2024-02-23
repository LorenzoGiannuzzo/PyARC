import pandas as pd


class DataNormalization:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def normalize_consumption(self):
        try:
            # Check if 'Consumption' column is present in the dataframe
            if 'Consumption' not in self.dataframe.columns:
                raise ValueError("The 'Consumption' column is not present in the dataframe.")

            # Check if the dataframe is not empty
            if self.dataframe.empty:
                raise ValueError("The dataframe is empty.")

            # Group the dataframe by User, Year, and Month, and calculate the maximum consumption for each group
            max_consumption_per_group = self.dataframe.groupby(['User', 'Year', 'Month'])['Consumption'].transform('max')

            # Create a new column 'Norm_consumption' with normalized values
            self.dataframe['Norm_consumption'] = self.dataframe['Consumption'] / max_consumption_per_group.replace(0, 1)
            # Replace 0 with 1 to avoid division by zero

            return self.dataframe

        except Exception as e:
            print(f"Error during data normalization: {e}")
            return None
