import copy
import os

import numpy as np
import torch

from utils.weight_utils import extract_weights, load_weights


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
