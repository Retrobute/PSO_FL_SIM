import matplotlib.pyplot as plt
from os import path
import numpy as np
def output_to_txt(txt_info , file_path) : 
    with open(file_path , "w") as file : 
        for info in txt_info :
            file.write(f"Iter : {info[0]} , Fitness : {info[1]:.4f} , \tTotal Processing Delay : {info[2]:.4f} , \tTotal Memory Score : {info[3]:.4f}\n")

def illustrate_plot(x , y , scatter_plot=False) : 
    plt.plot(x[1] , y[1] , color="blue" , zorder=1) 

    if scatter_plot : 
        plt.scatter(x[1] , y[1] , color="red" , zorder=2)

    plt.xlabel(x[0])
    plt.ylabel(y[0])
    plt.show()


def plot_tuple_curves(data):
    """
    Plots each tuple's elements in gray and overlays min, max, and average values.
    
    Parameters:
    data (list of tuples): A list where each tuple contains floating point numbers.
    """

    num_tuples = len(data)
    x = np.arange(num_tuples)
    
    # Convert list of tuples to a NumPy array for easier manipulation
    data_array = np.array(data, dtype=float)

    # Plot all tuple elements in gray
    for i in range(data_array.shape[1]):
        plt.plot(x, data_array[:, i], color='gray', alpha=0.5)
    
    # Compute min, max, and mean for each tuple
    min_vals = np.min(data_array, axis=1)
    max_vals = np.max(data_array, axis=1)
    avg_vals = np.mean(data_array, axis=1)
    
    # Overlay min, max, and average with distinct colors
    plt.plot(x, min_vals, color='green', label='Min')
    plt.plot(x, max_vals, color='red', label='Max')
    plt.plot(x, avg_vals, color='orange', label='Avg')
    
    plt.xlabel('Tuple Index')
    plt.ylabel('Values')
    plt.title('Tuple Elements with Min, Max, and Avg')
    plt.legend()
    plt.show()