from typing import Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


class TableLoader(Dataset):
    def __init__(self, X, y, device="cuda:0") -> None:
        super().__init__()

        self.X = X
        self.y = y
        self.device = device

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return torch.tensor(self.X[index]).float().to(self.device), torch.tensor(self.y[index]).float().to(self.device)


def build_loader(
    net: torch.nn.Sequential,
    loader: DataLoader,
    layer_idx: int,
    device: str,
    p_bar_desc: str,
    verbose: bool,
    shuffle: bool,
) -> DataLoader:
    """Формирование даталоадера

    Args:
        net (torch.nn.Sequential): объект сети
        dataloader (DataLoader): текущий обучающий или валидационный даталодер
        layer_idx (int): индекс слоя из которого считается новый даталодер
        net (torch.nn.Sequential): объект сети
        device (str): устройство вычисления. По умолчанию "cuda:0"
        p_bar_desc (ыек): текст, который будет отображаться в прогрессбаре
        verbose (bool): отображение прогрессбара tqdm. По умолчанию True
        shuffle (bool): замещение данных в лоадере

    Returns:
        Tuple[DataLoader, DataLoader]: обучающий и валидационный даталодеры на основе переданного индекса слоя
    """
    temp_x = []
    temp_y = []

    p_bar = tqdm(loader, disable=not verbose)
    p_bar.set_description(p_bar_desc)

    for x, y in p_bar:
        temp_x.append(net[(layer_idx - 1) * 2 : layer_idx * 2](x.to(device)).detach().cpu().numpy())
        temp_y.append(y.detach().cpu().numpy())

    new_train_dataloader = DataLoader(
        TableLoader(np.vstack(temp_x).copy(), np.vstack(temp_y).copy(), device=device),
        shuffle=shuffle,
        batch_size=loader.batch_size,
    )

    return new_train_dataloader


def create_intermediate_dataset(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    layer_idx: int,
    net: torch.nn.Sequential,
    device: str = "cuda:0",
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """_summary_

    Args:
        train_dataloader (DataLoader): текущий обучающий даталодер
        val_dataloader (DataLoader): текужий валидационный даталодер
        layer_idx (int): индекс слоя из которого считается новый даталодер
        net (torch.nn.Sequential): объект сети
        device (str): устройство вычисления. По умолчанию "cuda:0"
        verbose (bool): отображение прогрессбара tqdm. По умолчанию True

    Returns:
        Tuple[DataLoader, DataLoader]: обучающий и валидационный даталодеры на основе переданного индекса слоя
    """

    net = net.to(device)

    pbar_desc = f"Формирование обучающего даталодера (слой {layer_idx})"
    new_train_dataloader = build_loader(net, train_dataloader, layer_idx, device, pbar_desc, verbose, True)

    pbar_desc = f"Формирование валидационного даталодера (слой {layer_idx})"
    new_val_dataloader = build_loader(net, val_dataloader, layer_idx, device, pbar_desc, verbose, False)

    return new_train_dataloader, new_val_dataloader
