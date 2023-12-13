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
