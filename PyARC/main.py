# Import necessary modules
import os
from get_tou import CSVHandler as TouCSVHandler
from get_file import CSVHandler as DataCSVHandler
from data_preparation import DataFrameProcessor

# Define PyARC class
class PyARC:
    def train_model(self, data_path, tou_path):
        # Load data CSV
        data_handler = DataCSVHandler(data_path)
        data_handler.load_csv()
        data_dataframe = data_handler.get_data()

        # Load tou CSV
        tou_handler = TouCSVHandler(tou_path)
        tou_handler.load_csv()
        tou_dataframe = tou_handler.get_data()

        data = DataFrameProcessor(data_dataframe)
        data = data.process_dataframe()
#devo fare il check anche sul tou

        # Return the obtained DataFrames
        return data, tou_dataframe

# Check if the script is being run as the main program
if __name__ == "__main__":
    # Calculate full paths to the files
    data_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    data_file_path = os.path.join(data_file_directory, "default_data.csv")

    tou_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    tou_file_path = os.path.join(tou_file_directory, "default_tou.csv")

    # Create an instance of PyARC and call train_model
    pyarc_instance = PyARC()
    data, tou = pyarc_instance.train_model(data_file_path, tou_file_path)

