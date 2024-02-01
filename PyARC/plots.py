import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plots:
   def plot_norm_avg_cons(data):
       plt.figure(figsize=(12, 6))
       sns.lineplot(x="Hour", y="M_consumption", data=data, hue="User", ci=None)

       # Adding labels and title
       plt.xlabel("Hour")
       plt.ylabel("Normalized Average Monthly Consumption")
       plt.title("Normalized Average Monthly Consumption Profiles")
       plt.ylim(0, 1)

       # Save the plot as a .png file in the "plots" directory
       script_dir = os.path.dirname(__file__)  # Get the directory of the current script
       plots_dir = os.path.join(script_dir, "..", "plots")  # Navigate to the "plots" directory
       os.makedirs(plots_dir, exist_ok=True)  # Create the "plots" directory if it doesn't exist

       plt.savefig(os.path.join(plots_dir, "plot_image.png"))

