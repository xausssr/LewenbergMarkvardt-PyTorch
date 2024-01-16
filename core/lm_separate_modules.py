import time
from typing import Callable, List, Tuple, Union
import torch
from tqdm import tqdm

from core.gpu_lm import compute_jacobian
from utils.chekpoints import save_checkpoint, validate
from utils.weight_utils import get_weights_count, snap_weights, update_weights


# TODO сделать float16 обучение
def layer_separate_step(
    model: torch.nn.Module,
    output: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    mu: List[float],
    inner_steps: int = 10,
    demping_coef: float = 10,
    device: str = "cuda:0",
) -> List[float]:
    """Один шаг алгоритма Левенберга-Марквардта для разбитой на модули модели

    Модель должна содержать все свои параметры в полях model.group_x, где x - номер группы, например,
    model.group_0 - первая группа модели. Внутри группы допустима любая архитектура. Группировать таким
    образом, что бы совокупное число параметров (весов) группы <= 7000 (для 12Gb VRAM)

    Args:
        model (torch.nn.Module): модель, которая оптимизируется
        output (torch.Tensor): предыдущий отклик модели (на предыдущем шаге алгоритма)
        x (torch.Tensor): входные данные
        y (torch.Tensor): целевые метки
        loss_fn (Callable): функция ошибки (потерь)
        mu (List[float]): начальное значение коэффициента регуляризации - список рамером, равном количеству модулей
        val_loader (Union[torch.nn.DataLoader, None]) даталодер с валидационными данными
            (если есть - критерий останова на нем). Defaults to None.
        snap_folder (Union[str, None] = None) папка в которую складывать модели, если нет - не будет чекпоинтов
        inner_steps (int, optional): количество внутренних шагов. Defaults to 10.
        demping_coef (float, optional): коэффициент изменения регуляризации. Defaults to 10.
        device (str, optional): устройство вычислений. Defaults to "cuda:0".

    Returns:
        List[float]: список с коэффициентами регуляризации (mu) для каждой группы
    """

    last_loss = loss_fn(output, y)

    model_groups = [x for x in dict(model.named_modules()).keys() if "group_lm_" in x and "." not in x]
    group_sizes = [get_weights_count(getattr(model, x)) for x in model_groups]

    jac = compute_jacobian(model, x)
    error = (y - output).flatten()

    shift = 0
    for idx, (group_name, group_size) in enumerate(zip(model_groups, group_sizes)):
        group = getattr(model, group_name)
        snap = snap_weights(group)
        _temp_jac = jac[:, shift : group_size + shift]
        grad = _temp_jac.T @ error
        for i in range(inner_steps):
            approx_hess = _temp_jac.T @ _temp_jac + mu[idx] * torch.eye(_temp_jac.shape[1]).to(device)
            delta_w = (torch.inverse(approx_hess) @ grad).flatten()
            update_weights(group, snap + delta_w, device=device)
            output = model(x)
            curr_loss = loss_fn(output, y)

            if curr_loss <= last_loss:
                mu[idx] /= demping_coef
                break
            else:
                update_weights(group, snap, device=device)
                mu[idx] *= demping_coef

        shift += group_size

    return mu


# TODO привести все аргументы к одному виду (во всех подходах)
def train_levenberg_module_separate(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    min_error: float,
    max_epochs: int,
    val_loader: Union[torch.utils.data.DataLoader, None] = None,
    snap_folder: Union[str, None] = None,
    mu_init: float = 0.1,
    inner_steps: int = 10,
    demping_coef: int = 5,
    device="cuda:0",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Обучение с помощью алгоритма Левенберга-Марквардта на GPU для групп в моделе.
    Превосходит Adam по скорости, а Левенберга-Марквардта по использованию памяти - нечто среднее.
    Рекомендуется группы формировать по 7000 параметров, в зависимости от архитектуры - 2-4 слоёв в группе

    Args:
        model (torch.nn.Module): модель, которая оптимизируется
        x (torch.Tensor): входные данные
        y (torch.Tensor): целевые метки
        loss_fn (Callable): функция ошибки (потерь)
        val_loader (Union[torch.utils.data.DataLoader, None]) даталодер с валидационными данными
            (если есть - критерий останова на нем). Defaults to None.
        snap_folder (Union[str, None] = None) папка в которую складывать модели, если нет - не будет чекпоинтов
        mu_init (float, optional): начальное значение коэффициента регуляризации. Defaults to 10.
        min_error (float, optional): минимальная ошибка при которой останавливается обучение. Defaults to 1e-2.
        max_epochs (int, optional): максимльное число эпох при которых останавливается обучение. Defaults to 10.
        inner_steps (int, optional): количество внутренних шагов. Defaults to 10.
        demping_coef (float, optional): коэффициент изменения регуляризации. Defaults to 10.
        device (str, optional): устройство вычислений. Defaults to "cuda:0".

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            кортеж со списками истории обучения: ошибка, ошибка валидации, регуляризатор (mu), время вычислений на эпоху
    """

    model_groups = [x for x in dict(model.named_modules()).keys() if "group_lm_" in x and "." not in x]
    mu = [mu_init] * len(model_groups)
    output = model(x)

    loss_history = []
    val_loss_hisory = []
    if val_loader is not None:
        val_loss_hisory.append(validate(model, val_loader, loss_fn))
    loss_history.append(loss_fn(output, y).item())

    mu_history = {x: [] for x in model_groups}
    ep_time = [0]

    pbar = tqdm(range(max_epochs))
    pbar.set_description(f"MSE: {loss_history[-1]:.4f}")
    for ep in pbar:
        start_ep = time.time()
        mu = layer_separate_step(
            model, output, x, y, loss_fn, mu, inner_steps=inner_steps, demping_coef=demping_coef, device=device
        )
        output = model(x)
        loss = loss_fn(output, y).item()
        ep_time.append(time.time() - start_ep)

        loss_history.append(loss)
        for mu_idx, k in enumerate(mu_history.keys()):
            mu_history[k].append(mu[mu_idx])

        addition = ""
        if val_loader is not None:
            val_loss = validate(model, val_loader, loss_fn)
            val_loss_hisory.append(val_loss)
            addition = f" val: {val_loss:.3f}"

        if snap_folder is not None:
            save_checkpoint(model, snap_folder, ep)

        pbar.set_description(f"MSE: {loss:.3f}{addition}")
        pbar.update(1)
        if val_loader is None:
            if loss <= min_error:
                break
        else:
            if val_loss <= min_error:
                break

        for last_mu in mu:
            if last_mu > 1e29:
                pbar.set_description("Обучение остановлено (числовая нестабильность)")
                return loss_history, mu_history, ep_time

    return loss_history, val_loss_hisory, mu_history, ep_time
