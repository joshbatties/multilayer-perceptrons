import matplotlib.pyplot as plt
import pandas as pd

def plot_history(history, title="Training History"):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(title)
    plt.gca().set_ylim(0, 1)
    plt.show()

def plot_samples(X, y, class_names, n_rows=4, n_cols=10):
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()
