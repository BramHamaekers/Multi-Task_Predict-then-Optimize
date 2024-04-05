# file for plotting the data from the csv files in the different folders
import os
import csv
from matplotlib import ticker
import pandas as pd
import matplotlib.pyplot as plt
import re

i_range = [5, 15, 30, 60, 100, 150] # range of numer of items
n_range = [100]  # range of data
p_range = [10]  # range of features
deg_range = [4]  # range of degrees

# TODO CHANGE THIS TO THE EXPERIMENT TO RUN
experiment = i_range

# Define the wildcard pattern for filenames
filename_pattern = r"res_(?P<value>.*)\.csv"

# Function to read data from a CSV file and extract values
def read_data(filepath):
    with open(filepath) as f:
        data = pd.read_csv(f)
    value = re.match(filename_pattern, os.path.basename(filepath)).group("value")
    return data["Regret"], value # choose which column of data you want

def get_relative_regret(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        next(reader)
        relative_regrets = []
        for row in reader:
            relative_regret = float(row['Regret']) # choose which column of data you want
            relative_regrets.append(relative_regret)

    return relative_regrets

def get_average_regret(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        next(reader)
        average = 0
        rows = 0
        for row in reader:
            average += float(row['Regret']) # choose which column of data you want
            rows +=1
    return average/rows


def plot_data(data, title, x_label, y_label, out, min_val, max_val):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the data
    for strat, values in data.items():
        ax.plot(values, label=strat)

    # Set the title and labels
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)

    # Set the x-axis ticks
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(experiment))))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(experiment))

    # Set the y-axis limits
    ax.set_ylim(min_val, max_val)
    
    # Add a legend
    ax.legend(fontsize=18)

    # Save the plot to a file
    fig.savefig(out)

data = {}
strategy = ["comb", "gradnorm", "separated"]
tasks = ["multi1", "single1", "single2", "single3"]
loss_functs = ["spo", "pfyl"]
all_data = [] #gather all the data to be able to set a min and max for the y axis
avg_data = {} # gather the average of the different tasks
for loss in loss_functs:
    base_dir = "./res/3single1multi/" + loss
    avg_data[loss] = {strat: [0] * (len(experiment)) for strat in strategy}
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
                                    average_regret = get_average_regret(filepath)
                                    data[strat].append(average_regret)
                                    all_data.append(average_regret)
                                    #avg_data[loss][strat].append(average_regret)

        # Plot the 8 seperate graphs
        title = f"Average regret of the different strategies in a range of amount of Items for task={task}"
        x_label = "amount of Items"
        y_label = "Average regret over different tasks"
        # Sum the different values from the different tasks
        for strat in strategy: 
            for i in range(len(experiment)):
                avg_data[loss][strat][i] +=  data[strat][i]
        plot_data(data, title, x_label, y_label, str("./img/graphs/"+loss+"/"+task+".png"), min(all_data), max(all_data)) # choose where to save them
        
    # plot the average graph
    avg_title = f"Average regret in a range of amount of Items with loss_function={loss}"

    # calculate the average 
    for strat in strategy:
        for i in range(len(experiment)):
            avg_data[loss][strat][i] = avg_data[loss][strat][i] / 4
    plot_data(avg_data[loss], avg_title, x_label, y_label, str("./img/graphs/"+loss+"/average.png"), min(all_data), max(all_data))