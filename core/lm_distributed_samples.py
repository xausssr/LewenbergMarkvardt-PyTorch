import copy
import os
import shutil
import time
from typing import List, Tuple, Union

import numpy as np
import dask
import torch
from tqdm import tqdm
from core.chunk_utils import load_delayed
from utils.chekpoints import save_checkpoint, validate

from utils.weight_utils import extract_weights, get_weights_count, load_weights, snap_weights, update_weights


def distributed_jac(
    model: torch.nn.Module,
    x_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    temp_folder: str = "./temp",
) -> float:
    """Вычисление якобиана чанками по входным данным

    Args:
        model (torch.nn.Module): модель для которой считается якобиан
        x_loader (torch.utils.data.DataLoader): объект даталодера torch, batchsize будет соответсвовать размеру чанка
        loss_fn (torch.nn.Module): функция ошибки
        temp_folder (str, optional): папка для кеширования - создастся, если её нет в системе. Defaults to "./temp".

    Returns:
        float: _description_
    """
    w_count = get_weights_count(model)

    if w_count > 5000:
        raise NotImplementedError(f"Слишком большая сеть (число параметров = {w_count}). Данный случай не разработан")

    jac_model = copy.deepcopy(model)
    all_params, all_names = extract_weights(jac_model)
    load_weights(jac_model, all_names, all_params)
    os.makedirs(temp_folder, exist_ok=True)

    def param_as_input_func(model, x, param):
        load_weights(model, [name], [param])  # name is from the outer scope
        out = model(x)
        return out

    loss = []
    for idx, (x_batch, y_batch) in enumerate(x_loader):
        # Якобиан
        jac = []
        for i, (name, param) in enumerate(zip(all_names, all_params)):
            _t_jac = torch.autograd.functional.jacobian(
                lambda param: param_as_input_func(jac_model, x_batch, param),
                param,
                strict=True if i == 0 else False,
                vectorize=False if i == 0 else True,
            )
            jac.append(_t_jac.reshape((x_batch.shape[0], -1)).cpu().detach().numpy())
        jac = np.hstack(jac)
        np.save(os.path.join(temp_folder, f"jacobian_{idx:06d}"), jac)
        # Loss
        with torch.no_grad():
            output = model(x_batch)
            _t_loss = loss_fn(output, y_batch)
            loss.append(_t_loss.item())

            # Глобальная невязка
            error = (y_batch - output).cpu().detach().numpy()
            np.save(os.path.join(temp_folder, f"error_{idx:06d}"), error)

    del jac_model  # cleaning up
    return np.array(loss).mean()


def distributed_levenberg_step(
    model: torch.nn.Module,
    x_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    last_loss: float,
    temp_folder: str,
    mu: float,
    inner_steps: int = 10,
    demping_coef: int = 10,
    device: str = "cuda:0",
):
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
        _type_: _description_
    """
    # Чтение якобиана
    jac = load_delayed(os.path.join(temp_folder, "jacobian*"))
    error = load_delayed(os.path.join(temp_folder, "error*"))

    num_params = 0
    for p in model.parameters():
        num_params += np.prod(p.shape)

    grad = (jac.T @ error).compute()
    snap = snap_weights(model)

    for _ in range(inner_steps):
        approx_hess = (jac.T @ jac + mu * dask.array.eye(jac.shape[1])).astype(np.float32).compute()
        approx_hess = torch.linalg.inv(torch.tensor(approx_hess).float().to(device).float())
        delta_w = approx_hess @ torch.tensor(grad).float().to(device)
        delta_w = delta_w.flatten()

        # update weights
        update_weights(model, snap + delta_w, device=device)

        # get estimate
        curr_loss = []
        for x, y in x_loader:
            output = model(x)
            curr_loss.append(loss_fn(output, y).item())
        curr_loss = np.array(curr_loss).mean()

        # Adjust mu
        if curr_loss <= last_loss:
            return mu / demping_coef, curr_loss
        else:
            update_weights(model, snap, device=device)
            mu *= demping_coef

    return mu, curr_loss


def train_distributed_levenberg(
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
        model (torch.nn.Module): модель для которой считается якобиан
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
    loss_history = []
    val_loss_hisory = []
    if val_loader is not None:
        val_loss_hisory.append(validate(model, val_loader, loss_fn))

    mu_history = [mu_init]
    ep_time = [0]
    mu = mu_init
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
        mu, loss = distributed_levenberg_step(
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
        mu_history.append(mu)

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
