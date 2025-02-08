import matplotlib.pyplot as plt
from os import path

def output_to_txt(txt_info , file_path) : 
    with open(file_path , "w") as file : 
        for info in txt_info :
            file.write(f"Iter : {info[0]} , Fitness : {info[1]:.4f} , \tTotal Processing Delay : {info[2]:.4f} , \tTotal Memory Score : {info[3]:.4f}\n")

def illustrate_plot(x , y) : 
    plt.plot(x[1] , y[1])
    plt.xlabel(x[0])
    plt.ylabel(y[0])
    plt.show()