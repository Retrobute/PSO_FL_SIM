import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def illustrate_plot(csv_path, label, title, path, scatter_plot=False, M=4):
    # Load data from CSV
    df = pd.read_csv(csv_path)
    data = df.iloc[:M, 1].values  # Assuming first column is index, second column is data
    
    x = np.arange(1, len(data) + 1)
    
    plt.plot(x, data, color="blue", zorder=1, marker='o')
    
    if scatter_plot:
        plt.scatter(x, data, color="red", zorder=2)
    
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title(title)
    plt.grid(True)
    plt.savefig(path)
    plt.show()

def plot_csv_curves(csv_path, label, title, path, M=40):
    """
    Plots each column's elements with unique colors and markers.
    """
    df = pd.read_csv(csv_path)
    data_array = df.iloc[:M, 1:].values  # Exclude index column, limit to M iterations
    
    x = np.arange(M)
    colors = ['red', 'blue', 'orange']
    markers = ['o', 's', '^']  # Circle, Square, Triangle

    # Plot all tuple elements in gray
    for i in range(data_array.shape[1]):
        plt.plot(x, data_array[:, i], color='gray', alpha=0.5)
    
    # Compute min, max, and mean for each tuple
    min_vals = np.min(data_array, axis=1)
    max_vals = np.max(data_array, axis=1)
    avg_vals = np.mean(data_array, axis=1)
    
    # Overlay min, max, and average with distinct colors
    plt.plot(x, min_vals, color='green', marker = markers[0], label='Min TPD')
    plt.plot(x, max_vals, color='red',marker = markers[1], label='Max TPD')
    plt.plot(x, avg_vals, color='orange',marker = markers[2], label='Avg TPD')
    
    # for i in range(data_array.shape[1]):
    #     plt.plot(x, data_array[:, i], color=colors[i % len(colors)], marker=markers[i % len(markers)], alpha=0.7)
    
    
    plt.xlabel(label[0], fontsize=14)
    plt.ylabel(label[1], fontsize=14)
    # plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(title)
    plt.grid(True)
    plt.legend(fontsize = 14)
    plt.savefig(path)
    plt.show()

stepper = 1
DEPTH = 3
WIDTH = 5
pop_n = 5
scenario_file_number = WIDTH // stepper
scenario_folder_number = DEPTH                        
scenario_folder_name = f"scenario_case_{scenario_folder_number}"
particles_tpd_file = f"../results/pop_{pop_n}/{scenario_folder_name}/tpd_data_{scenario_file_number}.csv"
tpd_fig_path = f"../results/pop_{pop_n}/{scenario_folder_name}/new_tpd_{scenario_file_number}.pdf"

tpdl = ("PSO iteration" , "Normalized total processing delay")
tpdt = "Total Processing Delay Plot"

plot_csv_curves(particles_tpd_file,tpdl,tpdt,tpd_fig_path)