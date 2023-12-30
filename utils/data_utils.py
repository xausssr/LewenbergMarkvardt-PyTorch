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


# TODO добавть отключение tqdm
def create_intermediate_dataset(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    layer_idx: int,
    net: torch.nn.Sequential,
    device: str = "cuda:0",
) -> Tuple[DataLoader, DataLoader]:
    """_summary_

    Args:
        train_dataloader (DataLoader): текущий обучающий даталодер
        val_dataloader (DataLoader): текужий валидационный даталодер
        layer_idx (int): индекс слоя из которого считается новый даталодер
        net (torch.nn.Sequential): объект сети
        device (str): устройство вычисления. Defaults to "cuda:0"

    Returns:
        Tuple[DataLoader, DataLoader]: обучающий и валидационный даталодеры на основе переданного индекса слоя
    """

    temp_x = []
    temp_y = []

    net = net.to(device)
    p_bar = tqdm(train_dataloader)
    p_bar.set_description(f"Формирование обучающего даталодера (слой {layer_idx})")

    for x, y in p_bar:
        temp_x.append(net[(layer_idx - 1) * 2 : layer_idx * 2](x.to(device)).detach().cpu().numpy())
        temp_y.append(y.detach().cpu().numpy())
    new_train_dataloader = DataLoader(
        TableLoader(np.vstack(temp_x).copy(), np.vstack(temp_y).copy(), device=device),
        shuffle=True,
        batch_size=train_dataloader.batch_size,
    )

    temp_x = []
    temp_y = []
    p_bar = tqdm(val_dataloader)
    p_bar.set_description(f"Формирование валидационного даталодера (слой {layer_idx})")
    for x, y in p_bar:
        temp_x.append(net[(layer_idx - 1) * 2 : layer_idx * 2](x.to(device)).detach().cpu().numpy())
        temp_y.append(y.detach().cpu().numpy())
    new_val_dataloader = DataLoader(
        TableLoader(np.vstack(temp_x).copy(), np.vstack(temp_y).copy(), device=device),
        shuffle=False,
        batch_size=train_dataloader.batch_size,
    )
    return new_train_dataloader, new_val_dataloader
