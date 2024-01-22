import os
import pandas as pd

class CSVHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_csv(self):
        try:
            # Attempt to load the CSV file into a Pandas DataFrame
            self.data = pd.read_csv(self.file_path)
            print(f"File '{self.file_path}' loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
        except Exception as e:
            print(f"Error loading the file '{self.file_path}': {e}")

    def get_data(self):
        return self.data
