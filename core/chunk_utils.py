import glob
import dask
import numpy as np


def load_delayed(glob_selector: str) -> dask.array:
    """Загрузка чанка данных

    Args:
        glob_selector (str): селектор чанков (формат glob, например ./temp/error*.npy)

    Returns:
        dask.array: _description_
    """
    read = dask.delayed(np.load, pure=True)
    filenames = sorted(glob.glob(glob_selector))
    lazy_items = [read(path) for path in filenames]
    sample = lazy_items[0].compute()

    arrays = [dask.array.from_delayed(lazy_item, dtype=sample.dtype, shape=sample.shape) for lazy_item in lazy_items]

    return dask.array.concatenate(arrays, axis=0)
