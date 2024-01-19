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
            print("Load the CSV file first using the load_csv method.")

# Using the class
csv_handler = CSVHandler("default_data.csv")
csv_handler.load_csv()
csv_handler.display_data()
