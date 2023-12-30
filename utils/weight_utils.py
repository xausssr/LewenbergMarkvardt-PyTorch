import math
import os
import pickle
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor


def _del_nested_attr(obj: torch.nn.Module, names: List[str]) -> None:
    """Удаление атрибутов по списку именен. Например, для удаления obj.conv.weight
    используется `_del_nested_attr(obj, ['conv', 'weight'])`

    Args:
        obj (torch.nn.Module): объект модели
        names (List[str]): список имен для удаления
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def extract_weights(model: torch.nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """Удаляет все параметры (веса) из модели и преобразует их в обучный тензор.
    Веса должны быть повторно загружены (`load_weights`) если модель планируется
    использовать дальше.

    После применеия функции вызов `model.paramertes()` будет пустым

    Args:
        model (torch.nn.Module): модель, у которой необходимо преобразовать веса

    Returns:
        Tuple[Tuple[Tensor, ...], List[str]]: (список тензоров параметров модели и их имен)
    """

    orig_params = tuple(model.parameters())
    names = []
    for name, _ in list(model.named_parameters()):
        _del_nested_attr(model, name.split("."))
        names.append(name)

    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def _set_nested_attr(model: torch.nn.Module, names: List[str], value: Tensor) -> None:
    """Установка атрибута модели (регистрация тензора как параметра)

    Args:
        obj (torch.nn.Module): модель, у которой необходимо преобразовать веса
    """
    if len(names) == 1:
        setattr(model, names[0], value)
    else:
        _set_nested_attr(getattr(model, names[0]), names[1:], value)


def load_weights(model: torch.nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """Загрузка весов в модель (из тензора)

    Args:
        model (torch.nn.Module): модель, в которую необходимо загрузить веса
        names (List[str]): имя весов (атрибутов)
        params (Tuple[Tensor, ...]): тензора, значения которых загружаеются в модель (как веса)
    """
    for name, p in zip(names, params):
        _set_nested_attr(model, name.split("."), p)


def snap_weights(model: torch.nn.Module) -> torch.Tensor:
    """Получение значений весов модели (как тензоров)

    Args:
        model (torch.nn.Module): модель (инстанс pyTorch)

    Returns:
        torch.Tensor: веса модели (как плоский вектор)
    """
    params = []
    for parameter in model.parameters():
        params.append(parameter.flatten().clone())
    return torch.cat(params, dim=0)


def update_weights(model: torch.nn.Module, w: torch.Tensor, device: str = "cuda:0"):
    """Обновление весов модели

    Args:
        model (torch.nn.Module): модель (инстанс pyTorch)
        w (torch.Tensor): тензор весов (плоский вектор)
        device (str, optional): устройство вычислений. Defaults to "cuda:0".
    """
    idx = 0
    for param in model.parameters():
        param.data = w[idx : idx + np.prod(param.shape)].reshape(param.shape).to(device)
        idx += np.prod(param.shape)


def get_weights_count(model: torch.nn.Module) -> int:
    """Вычисляет количество параметров (весов) модели

    Args:
        model (torch.nn.Module): модель, для которой нужно вычислить число параметров

    Returns:
        int: число параметров модели
    """
    count = 0
    for p in model.parameters():
        count += np.prod(p.shape)

    return count


def get_layers_config(
    model: torch.nn.Module, in_size: int, max_params: int = 7000
) -> Tuple[List[List[int]], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Получение конфигурации сети для полносвязного персептрона - разбиение по слоям

    Args:
        model (torch.nn.Module): модель
        in_size (int): размерность входа
        max_params (int): максимальное количество параметров в слое

    Returns:
        Tuple[List[List[int]], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
            * список с количеством нейронов в каждом линейном слое (разбитый по максимуму параметров)
            * список ссылок на тензоры весов
            * список ссылок на тензоры смещений
    """

    layers = []

    weights = []
    biases = []
    for idx, param in enumerate(model.parameters()):
        if idx % 2 == 0:
            layers.append(param.shape[0])
            weights.append(param)
        else:
            biases.append(param)

    layers_config = []
    prev_size = in_size
    for idx, layer in enumerate(layers):
        total_params = prev_size * layer + layer
        if total_params < max_params:
            layer_split = [layer]
        else:
            max_neurons = math.floor(layer / (prev_size * layer / max_params))
            curr_neurons = 0
            layer_split = []
            while curr_neurons < layer:
                if layer - curr_neurons < max_neurons:
                    _temp_size = layer - curr_neurons
                else:
                    if max_neurons // 10 == 0:
                        raise RuntimeError(f"Слишком большой слой ({total_params}) - слой {idx}")
                    if max_neurons // 10 < 3:
                        low_border = 3
                    else:
                        low_border = max_neurons - max_neurons // 10
                    _temp_size = np.random.randint(low_border, max_neurons)
                layer_split.append(_temp_size)
                curr_neurons += _temp_size
        prev_size = layer
        layers_config.append(layer_split)
    return layers_config, weights, biases


def create_layer_from_parts(
    base_path: str,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter,
    layer_idx: int,
) -> None:
    """Собираем слой из кешей (для вычисления по блокам)

    Args:
        base_path (str): базовый путь до папки с кешированными данными
        weight (torch.nn.Parameter): объект весов сети, в который необходимо установить веса
        bias (torch.nn.Parameter): объект весов сети, в который необходимо установить смещения
        layer_idx (int): индекс слоя (0 - входной)
    """

    w = []
    b = []
    for folder in os.listdir(base_path):
        if "block" in folder:
            if int(folder.split("_")[1].split("_")[0]) == layer_idx:
                _history = pickle.load(open(os.path.join(base_path, folder, "train_history.bin"), "rb"))
                _best_idx = np.argmin(_history["val"])
                base_net = pickle.load(open(os.path.join(base_path, folder, "topology.bin"), "rb"))
                base_net.load_state_dict(
                    torch.load(os.path.join(base_path, folder, f"epoch_{_best_idx:06d}.pth"), map_location="cpu")
                )
                for idx, param in enumerate(base_net.parameters()):
                    if idx == 2:
                        break
                    if idx == 0:
                        w.append(param.data)
                    else:
                        b.append(param.data)
    w = torch.cat(w, dim=0)
    b = torch.cat(b, dim=0)
    weight.data = w.detach().cpu()
    bias.data = b.detach().cpu()
