import os
import pickle
from matplotlib import pylab as plt
import numpy as np


def plot_train_report_group(base_path: str) -> None:
    """Визуализация процесса обучения (для обучения группами)

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


def plot_train_report(base_path: str) -> None:
    """Визуализация процесса обучения (классического)

    Args:
        base_path (str): путь до результатов обучения
    """

    loss_history, val_history, mu_history, ep_time = tuple(pickle.load(open(base_path, "rb")).values())
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].plot(loss_history, label="Обучение")
    ax[0, 0].plot(val_history, label="Валидация")
    ax[0, 0].set_title("История обучения по эпохам")
    ax[0, 0].set_xlabel("Эпохи")
    ax[0, 0].set_ylabel("Среднеквадратичное отклонение")
    ax[0, 0].legend(loc="best")

    ax[0, 1].plot(mu_history)
    ax[0, 1].set_title("Mu по эпохам")
    ax[0, 1].set_xlabel("Эпохи")
    ax[0, 1].set_ylabel("Mu")
    ax[0, 1].set_yscale("log")

    ax[1, 0].plot(np.cumsum(ep_time), loss_history, label="Обучение")
    ax[1, 0].plot(np.cumsum(ep_time), val_history, label="Валидация")
    ax[1, 0].set_title("История обучения по времени")
    ax[1, 0].set_xlabel("Время, с")
    ax[1, 0].set_ylabel("Среднеквадратичное отклонение")
    ax[1, 0].legend(loc="best")

    ax[1, 1].plot(np.cumsum(ep_time), mu_history)
    ax[1, 1].set_title("Mu по времени")
    ax[1, 1].set_xlabel("Время, с")
    ax[1, 1].set_ylabel("Mu")
    ax[1, 1].set_yscale("log")
    plt.show()
