# Import necessary modules
import os

from PyARC import PyARC


# Check if the script is being run as the main program
if __name__ == "__main__":
    # Define full paths to the files

    data_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    data_file_path = os.path.join(data_file_directory, "default_data.csv")
    #
    tou_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Default Training Data")
    tou_file_path = os.path.join(tou_file_directory, "default_tou.csv")

    # Create an instance of PyARC and call train_model
    pyarc_instance = PyARC()
    #output = pyarc_instance.train_model(data_file_path, tou_file_path)

    pyarc_instance = PyARC()
    data = pyarc_instance.reconstruct_profiles()







