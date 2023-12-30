import os
import pickle
from matplotlib import pylab as plt
import numpy as np


def plot_train_history(base_path: str) -> None:
    """Визуализация процесса обучения

    Args:
        base_path (str): путь до папки с кешем обучения
    """

    train_dots = [[], []]
    val_dots = [[], []]

    for layer in range(4):
        for folder in os.listdir(base_path):
            if int(folder.split("_")[1].split("_")[0]) == layer:
                _history = pickle.load(open(os.path.join(base_path, folder, "train_history.bin"), "rb"))
                train_dots[0].append(layer + 1)
                train_dots[1].append(np.min(_history["loss"]))
                val_dots[0].append(layer + 1)
                val_dots[1].append(np.min(_history["val"]))

    plt.figure(figsize=(15, 5))
    plt.scatter(*train_dots)
    plt.scatter(*val_dots)
    plt.title("Лучшие (по валидации) ошибки обучения по слоям")
    plt.xlabel("Номер слоя")
    plt.ylabel("Среднеквадратичная ошибка")
    plt.show()
