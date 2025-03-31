import matplotlib.pyplot as plt
import numpy as np

def illustrate_plot(data , label , title , path , scatter_plot=False) : 
    x = np.arange(1, len(data) + 1)  

    plt.plot(x , data , color="blue" , zorder=1) 
    
    if scatter_plot : 
        plt.scatter(x , data , color="red" , zorder=2)

    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title(title)
    plt.savefig(path)
    plt.show()


def plot_tuple_curves(data , label , title , path):
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
    # for i in range(data_array.shape[1]):
    #     plt.plot(x, data_array[:, i], color='gray', alpha=0.5)
    
    # Compute min, max, and mean for each tuple
    min_vals = np.min(data_array, axis=1)
    max_vals = np.max(data_array, axis=1)
    avg_vals = np.mean(data_array, axis=1)
    
    # Overlay min, max, and average with distinct colors
    plt.plot(x, min_vals, color='green', label='Min')
    plt.plot(x, max_vals, color='red', label='Max')
    plt.plot(x, avg_vals, color='orange', label='Mean')
    
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.show()