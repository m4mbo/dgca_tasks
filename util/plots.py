import numpy as np
import matplotlib.pyplot as plt

def plot_sequence(y):
    """
    Plot a sequence y 
    """
    x = np.linspace(0, y.shape[1], y.shape[1])  
    plt.plot(x, y[0])  # plot the first sequence
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Sequence Plot")
    plt.show()