import copy
import time
from typing import Callable, List, Tuple
import torch
from tqdm import tqdm

from utils.weight_utils import extract_weights, get_weights_count, load_weights, snap_weights, update_weights


def compute_jacobian(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Вычисление якобиана для модели по входным данным

    Args:
        model (torch.nn.Module): модель, для которой необходимо вычислить якобиан (инстанс pyTorch)
        x (torch.Tensor): входные данные

    Returns:
        torch.Tensor: якобиан модели
    """

    jac_model = copy.deepcopy(model)
    all_params, all_names = extract_weights(jac_model)
    load_weights(jac_model, all_names, all_params)

    def param_as_input_func(model, x, param):
        load_weights(model, [name], [param])
        out = model(x)
        return out

    jac = []
    for i, (name, param) in enumerate(zip(all_names, all_params)):
        _t_jac = torch.autograd.functional.jacobian(
            lambda param: param_as_input_func(jac_model, x, param),
            param,
            strict=True if i == 0 else False,
            vectorize=False if i == 0 else True,
        )
        jac.append(_t_jac.reshape((x.shape[0], -1)))

    del jac_model  # cleaning up
    return torch.cat(jac, dim=-1)


def levenberg_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    mu: float,
    inner_steps: int = 10,
    demping_coef: float = 10,
    device: str = "cuda:0",
) -> Tuple[float, float]:
    """Один шаг алгоритма Левенберга-Марквардта

    Args:
        model (torch.nn.Module): модель, которая оптимизируется
        x (torch.Tensor): входные данные
        y (torch.Tensor): целевые метки
        loss_fn (Callable): функция ошибки (потерь)
        mu (float): начальное значение коэффициента регуляризации
        inner_steps (int, optional): количество внутренних шагов. Defaults to 10.
        demping_coef (float, optional): коэффициент изменения регуляризации. Defaults to 10.
        device (str, optional): устройство вычислений. Defaults to "cuda:0".

    Returns:
        Tuple[float, float]: кортеж с коэффициентом регуляризации (mu) и ошибка после шага алгоритма
    """
    w_count = get_weights_count(model)
    obj_count = x.shape[0]

    if obj_count > 5_000:
        if w_count > 5000:
            raise RuntimeError(
                f"Слишком много данных для вычисления на GPU ({obj_count}x{w_count}), используйте "
                "распределенный вариант"
            )
        else:
            raise RuntimeError(
                f"Слишком много данных для вычисления на GPU ({obj_count}x{w_count})"
                ", используйте core.lm_distributed_samples"
            )

    output = model(x)
    last_loss = loss_fn(output, y)
    jac = compute_jacobian(model, x)

    # TODO разобраться с многовыходным случаем, пока наивно
    if output.ndim > 2:
        raise NotImplementedError("Для многомерного выхода - не имплементированно")
    if output.shape[-1] > 1:
        error = torch.argmax(y, dim=1) - torch.argmax(output, dim=1)
    else:
        error = y - output
    grad = jac.T @ error
    snap = snap_weights(model)

    for i in range(inner_steps):
        approx_hess = jac.T @ jac + mu * torch.eye(jac.shape[1]).to(device)
        delta_w = (torch.inverse(approx_hess) @ grad).flatten()
        update_weights(model, snap + delta_w, device=device)
        output = model(x)
        curr_loss = loss_fn(output, y)
        if curr_loss <= last_loss:
            return mu / demping_coef, curr_loss.item()
        else:
            update_weights(model, snap, device=device)
            mu *= demping_coef

    return mu, curr_loss.item()


def train_levenberg(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    mu_init: float = 10,
    min_error: float = 1e-2,
    max_epochs: int = 10,
    inner_steps: int = 10,
    demping_coef: float = 10,
    device: str = "cuda:0",
) -> Tuple[List[float], List[float], List[float]]:
    """Обучение с помощью алгоритма Левенберга-Марквардта на GPU (все матрицы в памяти, нет кеширования на диск)

    Args:
        model (torch.nn.Module): модель, которая оптимизируется
        x (torch.Tensor): входные данные
        y (torch.Tensor): целевые метки
        loss_fn (Callable): функция ошибки (потерь)
        mu_init (float, optional): начальное значение коэффициента регуляризации. Defaults to 10.
        min_error (float, optional): минимальная ошибка при которой останавливается обучение. Defaults to 1e-2.
        max_epochs (int, optional): максимльное число эпох при которых останавливается обучение. Defaults to 10.
        inner_steps (int, optional): количество внутренних шагов. Defaults to 10.
        demping_coef (float, optional): коэффициент изменения регуляризации. Defaults to 10.
        device (str, optional): устройство вычислений. Defaults to "cuda:0".

    Returns:
        Tuple[List[float], List[float], List[float]]: кортеж со списками истории обучения: ошибка, регуляризатор (mu) и
            время вычислений на эпоху
    """
    loss_history = []
    mu_history = [mu_init]
    ep_time = [0]

    mu = mu_init
    output = model(x)
    loss_history.append(loss_fn(output, y).item())

    pbar = tqdm(range(max_epochs))
    pbar.set_description(f"Loss: {loss_history[-1]:.4f}")
    for _ in pbar:
        start_ep = time.time()
        mu, loss = levenberg_step(
            model, x, y, loss_fn, mu, inner_steps=inner_steps, demping_coef=demping_coef, device=device
        )
        ep_time.append(time.time() - start_ep)
        loss_history.append(loss)
        mu_history.append(mu)
        pbar.set_description(f"Loss: {loss_history[-1]:.4f}")
        if loss <= min_error:
            break
    return loss_history, mu_history, ep_time
