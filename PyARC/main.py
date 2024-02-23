# Import necessary libraries
import os
from PyARC import PyARC

# Class to manage PyARC models


class ModelManager:

    def __init__(self):
        # Initialize PyARC instance
        self.pyarc_instance = PyARC()

    # Method to use a pre-trained model
    def use_pretrained_model(self):
        # Reconstruct profiles using the pre-trained model
        self.pyarc_instance.reconstruct_profiles()

    # Method to train a new model
    def train_new_model(self):
        # Set paths for input training data
        data_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Input Training Data")
        data_file_path = os.path.join(data_file_directory, "train_data.csv")

        tou_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Input Training Data")
        tou_file_path = os.path.join(tou_file_directory, "train_tou.csv")

        # Train a new model using PyARC
        self.pyarc_instance.train_model(data_file_path, tou_file_path)

    # Method to use a user-trained model
    def use_user_trained_model(self):
        # Use the user-trained model
        self.pyarc_instance.user_trained_model()


# Function to start the program
def start_program():
    print("Welcome to PyARC! What would you like to do?")
    print("1. Reconstruct Residential Aggregate Electrical Load Profiles using the pre-trained model")
    print("2. Train a new model")
    print("3. Reconstruct Residential Aggregate Electrical Load Profiles using the user-trained model")

    # Get user input for the desired action
    choice = input("Enter the number corresponding to the desired action: ")

    # Create an instance of ModelManager
    model_manager = ModelManager()

    # Execute the chosen action
    if choice.strip() == "1":
        model_manager.use_pretrained_model()
        print("Pre-trained model successfully used to reconstruct profiles.")
    elif choice.strip() == "2":
        model_manager.train_new_model()
        print("New model successfully trained.")
    elif choice.strip() == "3":
        model_manager.use_user_trained_model()
        print("User-trained model successfully used to reconstruct profiles.")
    else:
        print("Invalid choice. Please enter a valid number.")


# Entry point for the script
if __name__ == "__main__":
    # Start the program
    start_program()
