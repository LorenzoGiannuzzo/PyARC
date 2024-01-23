# Import necessary modules
import os
from get_tou import CSVHandler as TouCSVHandler
from get_file import CSVHandler as DataCSVHandler
from data_preparation import DataFrameProcessor
from data_preprocessing import DataPreprocessing

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

        DataFrameProcessor.check_tou(tou_dataframe)

        data_summer, data_winter, data_spring, data_autumn = DataFrameProcessor.data_subset(data)

        # Creating a DataPreprocessing object
        data_processor = DataPreprocessing(data)

        # Calling the get_negative_values function
        corrected_data = data_processor.get_negative_values()
        corrected_data = DataPreprocessing.replace_max_daily_zero_consumption(corrected_data)

        corrected_data = DataPreprocessing.process_outliers(corrected_data)

        # Return the obtained DataFrames
        return corrected_data

# Check if the script is being run as the main program
if __name__ == "__main__":
    # Calculate full paths to the files
    data_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    data_file_path = os.path.join(data_file_directory, "default_data.csv")

    tou_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    tou_file_path = os.path.join(tou_file_directory, "default_tou.csv")

    # Create an instance of PyARC and call train_model
    pyarc_instance = PyARC()
    corrected_data = pyarc_instance.train_model(data_file_path, tou_file_path)

