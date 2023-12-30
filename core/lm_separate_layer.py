import os
import pickle
import shutil
from typing import List, Tuple
from aiohttp_retry import Union
import torch
from core.lm_distributed_samples import train_distributed_levenberg
from utils.data_utils import create_intermediate_dataset
from utils.visualisation_utils import plot_train_history

from utils.weight_utils import create_layer_from_parts, get_layers_config


def train_layer_separeate(
    model: torch.nn.Sequential,
    x_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    val_loader: Union[torch.utils.data.DataLoader, None] = None,
    mu_init: int = 10,
    min_error: float = 1e-2,
    max_epochs: int = 10,
    inner_steps: int = 10,
    demping_coef: float = 10,
    device: str = "cuda:0",
    temp_folder: str = "./temp",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
        Распределенное вычисление (по слоям - каждый слой разбивается на отрезки по 5000 нейронов)
        алгоритма Левенберга-Марквардта.

    Args:
        model (torch.nn.Sequential): модель для которой считается якобиан (поддерживается только Sequential)
        x_loader (torch.utils.data.DataLoader): объект даталодера torch, batchsize будет соответсвовать размеру чанка
        loss_fn (torch.nn.Module): функция ошибки
        val_loader (Union[torch.utils.data.DataLoader, None]) даталодер с валидационными данными
            (если есть - критерий останова на нем). Defaults to None.
        mu_init (int, optional): начальное значение коэффициента регуляризации. Defaults to 10.
        min_error (float, optional): минимальная ошибка при которой останавливается обучение. Defaults to 1e-2.
        max_epochs (int, optional): максимльное число эпох при которых останавливается обучение. Defaults to 10.
        inner_steps (int, optional): количество внутренних шагов. Defaults to 10.
        demping_coef (float, optional): коэффициент изменения регуляризации. Defaults to 10.
        device (str, optional): устройство вычислений. Defaults to "cuda:0".
        temp_folder (str, optional): папка с кешами. Defaults to "./temp".

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            кортеж со списками истории обучения: ошибка, ошибка валидации, регуляризатор (mu), время вычислений на эпоху

    Raises:
        NotImplementedError: вызывается при передаче в качестве модели torch.nn.Module, поддерживается только Sequential
    """

    if not isinstance(model, torch.nn.Sequential):
        raise NotImplementedError(f"Поддерживается только torch.nn.Sequential (передан {type(model)})!")

    in_size = next(iter(x_loader))[0].shape[1]
    out_size = next(iter(x_loader))[1].shape[1]

    if out_size == 1:
        last_act = torch.nn.Sigmoid()
    else:
        last_act = torch.nn.Softmax(dim=1)

    separate_layers, weights, biases = get_layers_config(model=model, in_size=in_size, max_params=5000)

    os.makedirs(os.path.join(temp_folder, "inner_temp"), exist_ok=True)

    prev_size = in_size
    for layer_idx, layer in enumerate(separate_layers):
        for group_idx, group in enumerate(layer):
            os.makedirs(os.path.join(temp_folder, f"block_{layer_idx}_{group_idx}"), exist_ok=True)
            snap_folder = os.path.join(temp_folder, f"block_{layer_idx}_{group_idx}")
            if layer_idx < len(separate_layers) - 1:
                temp_net = torch.nn.Sequential(
                    torch.nn.Linear(prev_size, group), torch.nn.Sigmoid(), torch.nn.Linear(group, out_size), last_act
                )
            else:
                temp_net = torch.nn.Sequential(torch.nn.Linear(prev_size, out_size), last_act)
            temp_net = temp_net.to(device)
            loss_history, val_history, mu_history, ep_time = train_distributed_levenberg(
                temp_net,
                x_loader,
                loss_fn,
                val_loader,
                snap_folder=snap_folder,
                mu_init=mu_init,
                inner_steps=inner_steps,
                min_error=min_error,
                max_epochs=max_epochs,
                demping_coef=demping_coef,
                device=device,
                temp_folder=os.path.join(temp_folder, "inner_temp"),
            )

            pickle.dump(
                {"loss": loss_history, "val": val_history, "mu": mu_history, "time": ep_time},
                open(os.path.join(snap_folder, "train_history.bin"), "wb"),
            )
            pickle.dump(temp_net, open(os.path.join(snap_folder, "topology.bin"), "wb"))
        create_layer_from_parts(base_path=temp_folder, weight=weights[0], bias=biases[0], layer_idx=0)
        if layer_idx < len(separate_layers) - 1:
            x_loader, val_loader = create_intermediate_dataset(
                train_dataloader=x_loader, val_dataloader=val_loader, layer_idx=layer_idx + 1, net=model, device=device
            )
        prev_size = sum(layer)

    plot_train_history(temp_folder)
    shutil.rmtree(temp_folder)
    return
