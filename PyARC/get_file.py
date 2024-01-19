import os
import pandas as pd

class CSVHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_csv(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"File '{self.file_path}' loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
        except Exception as e:
            print(f"Error loading the file '{self.file_path}': {e}")

    def display_data(self):
        if self.data is not None:
            print(self.data)
        else:
            print("No data loaded. Call load_csv first.")

# Calcola il percorso completo al file
file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
file_path = os.path.join(file_directory, "default_data.csv")

# Using the class
csv_handler = CSVHandler(file_path)

# Chiamata a load_csv prima di display_data
csv_handler.load_csv()
csv_handler.display_data()

