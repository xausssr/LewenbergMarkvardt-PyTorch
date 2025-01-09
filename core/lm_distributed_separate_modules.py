import os
import shutil
import time
from typing import List, Tuple, Union
import numpy as np
import torch
import dask
from tqdm import tqdm

from core.chunk_utils import load_delayed
from core.lm_distributed_samples import distributed_jac
from utils.chekpoints import save_checkpoint, validate
from utils.weight_utils import get_weights_count, snap_weights, update_weights


# TODO Рефакторинг в нормальный вид
# TODO добавить правильное логгированию mu
def distributed_modules_levenberg_step(
    model: torch.nn.Module,
    x_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    last_loss: float,
    temp_folder: str,
    mu: List[float],
    inner_steps: int = 10,
    demping_coef: int = 10,
    device: str = "cuda:0",
) -> Tuple[List[float], float]:
    """Вычисление шага алгоритма Левенберга-Марквардта

    Args:
        model (torch.nn.Module): модель для которой считается якобиан
        x_loader (torch.utils.data.DataLoader): объект даталодера torch, batchsize будет соответсвовать размеру чанка
        loss_fn (torch.nn.Module): функция ошибки обучения
        last_loss (float): последнее значение ошибки обучения
        temp_folder (str): путь до папки с кешем
        mu (float): начальное значение коэффициента регуляризации
        inner_steps (int, optional): количество внутренних шагов. Defaults to 10.
        demping_coef (float, optional): коэффициент изменения регуляризации. Defaults to 10.
        device (str, optional): устройство вычислений. Defaults to "cuda:0".

    Raises:
        RuntimeError: _description_

    Returns:
        Tuple[List[float], float]: текущее значение mu (по группам модулей) и ошибки
    """
    # Формирование групп модулей
    model_groups = [x for x in dict(model.named_modules()).keys() if "group_lm_" in x and "." not in x]
    group_sizes = [get_weights_count(getattr(model, x)) for x in model_groups]

    # Чтение якобиана
    jac = load_delayed(os.path.join(temp_folder, "jacobian*"))
    error = load_delayed(os.path.join(temp_folder, "error*"))

    snap = snap_weights(model)

    shift = 0
    for idx, (group_name, group_size) in enumerate(zip(model_groups, group_sizes)):
        group = getattr(model, group_name)
        snap = snap_weights(group)
        _temp_jac = jac[:, shift : group_size + shift]
        grad = (_temp_jac.T @ error).compute()
        for _ in range(inner_steps):
            approx_hess = (
                (_temp_jac.T @ _temp_jac + mu[idx] * dask.array.eye(_temp_jac.shape[1])).astype(np.float32).compute()
            )
            approx_hess = torch.linalg.inv(torch.tensor(approx_hess).float().to(device).float())
            delta_w = approx_hess @ torch.tensor(grad).float().to(device)
            delta_w = delta_w.flatten()

            # update weights
            update_weights(group, snap + delta_w, device=device)

            # get estimate
            curr_loss = []
            for x, y in x_loader:
                output = model(x)
                curr_loss.append(loss_fn(output, y).item())
            curr_loss = np.array(curr_loss).mean()

            # Adjust mu
            if curr_loss <= last_loss:
                mu[idx] /= demping_coef
                break
            else:
                update_weights(group, snap, device=device)
                mu[idx] *= demping_coef

        shift += group_size

    return mu, curr_loss


def train_distributed_modules_levenberg(
    model: torch.nn.Module,
    x_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    val_loader: Union[torch.utils.data.DataLoader, None] = None,
    snap_folder: Union[str, None] = None,
    mu_init: int = 10,
    min_error: float = 1e-2,
    max_epochs: int = 10,
    inner_steps: int = 10,
    demping_coef: float = 10,
    device: str = "cuda:0",
    temp_folder: str = "./temp",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Распределенное вычисление (по обучающим примерам) алгоритма Левенберга-Марквардта

    Args:
        model (torch.nn.Module): модель, которая оптимизируется
        x_loader (torch.utils.data.DataLoader): объект даталодера torch, batchsize будет соответсвовать размеру чанка
        loss_fn (torch.nn.Module): функция ошибки
        val_loader (Union[torch.utils.data.DataLoader, None]) даталодер с валидационными данными
            (если есть - критерий останова на нем). Defaults to None.
        snap_folder (Union[str, None] = None) папка в которую складывать модели, если нет - не будет чекпоинтов
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
    """
    model_groups = [x for x in dict(model.named_modules()).keys() if "group_lm_" in x and "." not in x]
    mu = [mu_init] * len(model_groups)

    mu_history = {x: [] for x in model_groups}
    loss_history = []
    val_loss_hisory = []
    if val_loader is not None:
        val_loss_hisory.append(validate(model, val_loader, loss_fn))

    ep_time = [0]
    pbar = tqdm(range(max_epochs))

    os.makedirs(temp_folder, exist_ok=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    for ep in range(max_epochs):
        start_ep = time.time()

        # Считаем чанки якобиана
        loss = distributed_jac(model, x_loader, loss_fn, temp_folder=temp_folder)
        if ep == 0:
            loss_history.append(loss)
            pbar.set_description(f"MSE: {loss:.3f}")

        # Шаг алгоритма
        mu, loss = distributed_modules_levenberg_step(
            model,
            x_loader,
            loss_fn=loss_fn,
            last_loss=loss,
            temp_folder=temp_folder,
            mu=mu,
            inner_steps=inner_steps,
            demping_coef=demping_coef,
            device=device,
        )
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

        pbar.update(1)
        pbar.set_description(f"MSE: {loss:.3f}{addition}")

        if val_loader is None:
            if loss <= min_error:
                break
        else:
            if val_loss <= min_error:
                break

    shutil.rmtree(temp_folder)
    return loss_history, val_loss_hisory, mu_history, ep_time
