# Import necessary modules
import pandas as pd

# Define CSVHandler class
class CSVHandler:
    def __init__(self, file_path):
        # Initialize CSVHandler with a file path
        self.file_path = file_path
        self.data = None

    def load_csv(self):
        # Attempt to load the CSV file into a Pandas DataFrame
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"File '{self.file_path}' loaded successfully.")
        except FileNotFoundError:
            # Handle FileNotFoundError, print an error message
            print(f"Error: The file '{self.file_path}' was not found.")
        except Exception as e:
            # Handle other exceptions, print an error message with details
            print(f"Error loading the file '{self.file_path}': {e}")

    def get_data(self):
        # Return the loaded DataFrame
        return self.data