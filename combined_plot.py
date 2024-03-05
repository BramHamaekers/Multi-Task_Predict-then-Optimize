# file for plotting the data from the csv files in the different folders
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re

i_range = [30] # range of numer of items
n_range = [25, 50, 100, 250, 500, 1000]  # range of data
p_range = [10]  # range of features
deg_range = [4]  # range of degrees

# Define the wildcard pattern for filenames
filename_pattern = r"res_(?P<value>.*)\.csv"

# Function to read data from a CSV file and extract values
def read_data(filepath):
    with open(filepath) as f:
        data = pd.read_csv(f)
    value = re.match(filename_pattern, os.path.basename(filepath)).group("value")
    return data["Relative Regret"], value # choose which column of data you want

def get_relative_regret(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        next(reader)
        relative_regrets = []
        for row in reader:
            relative_regret = float(row['Relative Regret']) # choose which column of data you want
            relative_regrets.append(relative_regret)

    return relative_regrets

def get_average_regret(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        next(reader)
        average = 0
        rows = 0
        for row in reader:
            average += float(row['Relative Regret']) # choose which column of data you want
            rows +=1
    return average/rows


def plot_data(data, title, x_label, y_label, out):
    plt.figure(figsize=(10, 6))
    for key in data.keys():
        plt.plot(n_range, data[key], label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out)

data = {}
strategy = ["comb", "gradnorm", "separated"]
tasks = ["multi1", "single1", "single2", "single3"]
loss_functs = ["spo", "pfyl"]

for loss in loss_functs:
    base_dir = "./res/3single1multi/" + loss
    for task in tasks:
        for strat in strategy:
            data[strat] = []
            for i in i_range:
                for n in n_range: 
                    for p in p_range:
                        for deg in deg_range:
                            folder_name = f"i{i}n{n}p{p}deg{deg}"
                            folder_path = os.path.join(base_dir, folder_name)
                            if not os.path.exists(folder_path):
                                print("folder doesn't exist: " + folder_path)
                                continue
                            for filename in os.listdir(folder_path):
                                if re.match(filename_pattern, filename) and strat in filename and task in filename:
                                    filepath = os.path.join(folder_path, filename)
                                    data[strat].append(get_average_regret(filepath))

        title = f"Average Relative Regret of the different strategies in a range of data sizes for task={task}"
        x_label = "Number of Data"
        y_label = "Average Relative Regret"

        plot_data(data, title, x_label, y_label, str("./img/graphs/"+loss+"/"+task+".png")) # choose where to save them