import os
import torch


def validate(net: torch.nn.Module, val_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module) -> float:
    """Валидация сети

    Args:
        net (torch.nn.Module): сеть
        val_loader (torch.utils.data.DataLoader): даталодер для валидационных данных
        loss_fn (torch.nn.Module): функция потерь

    Returns:
        float: ошибка валидации
    """

    loss = 0
    for x, y in val_loader:
        out = net(x)
        _loss_item = loss_fn(out, y).item()
        loss += _loss_item
    return loss / len(val_loader)


def save_checkpoint(net: torch.nn.Module, folder: str, epoch: int) -> None:
    """Сохранение чекпоинта обучения

    Args:
        net (torch.nn.Module): сеть, которая сохраняется
        folder (str): папка для сохранения
        epoch (int): текущая эпоха
    """

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    torch.save(net.state_dict(), os.path.join(folder, f"epoch_{epoch:06d}.pth"))
