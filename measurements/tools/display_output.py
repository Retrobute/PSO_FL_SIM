from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

def show_plot(data, label, title, path):
    x = np.arange(1, len(data) + 1)

    plt.figure()
    plt.plot(x, data, color="blue", zorder=1)
    plt.scatter(x, data, color="blue", marker="o", zorder=2)

    legend_element = Line2D([0], [0], color='blue', marker='o', label="Swarm's Gbest")
    
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.xlabel(label[0], fontsize=14)
    plt.ylabel(label[1], fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(handles=[legend_element], fontsize=12)

    plt.savefig(path, dpi=300)
    plt.show()

def plot_tuple_curves(data, label, title, path):
    num_tuples = len(data)
    x = np.arange(num_tuples)
    
    data_array = np.array(data, dtype=float)

    _, ax = plt.subplots()  
    for i in range(data_array.shape[1]):
        ax.plot(x, data_array[:, i], color='gray', alpha=0.5)
    
    min_vals = np.min(data_array, axis=1)
    max_vals = np.max(data_array, axis=1)
    avg_vals = np.mean(data_array, axis=1)

    ax.plot(x, min_vals, color='green')
    ax.scatter(x, min_vals, marker="o", color='green')

    ax.plot(x, max_vals, color='red')
    ax.scatter(x, max_vals, marker="s", color='red')

    ax.plot(x, avg_vals, color='orange')
    ax.scatter(x, avg_vals, marker="^", color='orange')

    legend_elements = [
        Line2D([0], [0], color='green', marker='o', label='Min TPD'),
        Line2D([0], [0], color='red', marker='s', label='Max TPD'),
        Line2D([0], [0], color='orange', marker='^', label='Mean TPD')
    ]

    ax.grid(True, which='both', color='gray', linewidth=0.5)
    ax.set_xlabel(label[0], fontsize=12) 
    ax.set_ylabel(label[1], fontsize=12)  
    ax.set_title(title, fontsize=16)  
    ax.legend(handles=legend_elements)
    plt.savefig(path, dpi=300)
    plt.show()


def histogram_plot(data, path, label=("Bins", "Frequency")):
    plt.figure()
    _, _, _ = plt.hist(data, color="blue", alpha=0.7, edgecolor="black", zorder=1)

    legend_element = Line2D([0], [0], color='blue', marker='s', linestyle='None', label="Data Frequency")

    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.xlabel(label[0], fontsize=12)
    plt.ylabel(label[1], fontsize=12)
    plt.legend(handles=[legend_element])

    plt.savefig(path, dpi=300)
    plt.show()
