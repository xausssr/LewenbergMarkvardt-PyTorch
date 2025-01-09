# Набор алгоритмов обучения, основанный на гессиане

## Общие утилиты

1. Для получения количества параметров (весов) модели использовать функцию `utils.weight_util.get_weights_count`.
2. Подготовлен простой класс датасета (`torch.utils.data.Dataset`) для загрузки табличных данных `utils.data_utils.TableLoader`:

```python
import sys
import numpy as np

sys.path.append("<...>/LewenbergMarkvardt/lib")
from utils.data_utils import TableLoader

x = np.random.randn(5000, 100)
y = np.eye(2)[np.random.randint(0, 2, size=5000)] # OneHot для бинарной переменной

dataset = TableLoader(x, y, device='cpu')
```

3. Визуализация процесса обучения (из дампа pickle) `utils.visualisation_utils.plot_train_report` - работает с `core.gpu_lm.train_levenberg` и
`core.lm_distributed_samples`

```python
loss_history, val_history, mu_history, ep_time = train_levenberg
pickle.dump(
    {"loss": loss_history, "val": val_history, "mu": mu_history, "time": ep_time},
    open("<path>", "wb"), 
)
plot_train_report("<path>")
```



## Классический Левенберг-Марквардт (для любых сетей)

Реализован алгоритм для обучения любых архитектур без кеширования в файловой системе. Ограничения для GPU 12Gb: 

* число весов (параметров) сети `<=7000`,
* [число обучающих примеров  $\times$ число выходных нейронов] `<=7000`.

Данный алгоритм может применяться для решения задач с относительно небольшим количеством данных, небольшими моделями и
необходимостью быстро переобучать модели.

Использование:

```python
import sys
import torch

sys.path.append("<...>/LewenbergMarkvardt/lib")

from core.gpu_lm import train_levenberg
from utils.visualisation_utils import plot_train_report

# входные данные
x = ... # torch.FloatTensor
y = ... # torch.FloatTensor

# Модель - может быть torch.nn.Sequential или torch.nn.Module -- ограничение только на количество параметров
# модель предварительно поместить на устройство вычислений
model = torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 50), 
    torch.nn.Sigmoid()
    torch.nn.Linear(50, y.shape[1]),
    torch.nn.Softmax(dim=1)
).cuda()

# Функкция ошибки, рекомендована MSELoss (исходя из теоретических соображений оптимизации методами 2 порядка)
loss_fn = torch.nn.MSELoss()

loss_history, val_history, mu_history, ep_time = train_levenberg(
    model,
    x,
    y,
    loss_fn,
    val_loader=None,
    snap_folder=None,
    mu_init=10,
    min_error=1e-2,
    max_epochs=10,
    inner_steps=10,
    demping_coef=10,
    device="cuda:0",
)

# Сохраняем все метрики
pickle.dump(
    {"loss": loss_history, "val": val_history, "mu": mu_history, "time": ep_time},
    open("test_run.pkl", "wb"), 
)

# Визуализируем
plot_train_report("test_run.pkl")

# Объект model с настроенными весами - если не была указан `snap_folder`, нужно сохранить веса отдельно
torch.save(model.state_dict, 'weights.pkl')
```

**Описание аргументов:**
   
* `model (torch.nn.Module)`: модель, которая оптимизируется
* `x (torch.Tensor)`: входные данные
* `y (torch.Tensor)`: целевые метки
* `loss_fn (Callable)`: функция ошибки (потерь)
* `val_loader (Union[torch.utils.data.DataLoader, None])`: даталодер с валидационными данными (если есть - `min_error` счиатется на нем). `Defaults to None.`
* `snap_folder (Union[str, None] = None)`: папка в которую складывать модели, если нет - не будет чекпоинтов
* `mu_init (float, optional)`: начальное значение коэффициента регуляризации. `Defaults to 10.`
* `min_error (float, optional)`: минимальная ошибка при которой останавливается обучение. `Defaults to 1e-2.`
* `max_epochs (int, optional)`: максимльное число эпох при которых останавливается обучение. `Defaults to 10.`
* `inner_steps (int, optional)`: количество внутренних шагов. `Defaults to 10.`
* `demping_coef (float, optional)`: коэффициент изменения регуляризации. `Defaults to 10.`
* `device (str, optional)`: устройство вычислений. `Defaults to "cuda:0".`

**Возвращает:**
`Tuple[List[float], List[float], List[float], List[float]]`: кортеж со списками истории обучения: ошибка, ошибка валидации, регуляризатор (mu), время вычислений на эпоху

Подробнее об алгоритме в разрезе нейронных сетей:
* [А. Голубинский, А. Толстых, раздел 3 (статья автора репозитория)](https://www.elibrary.ru/download/elibrary_45594824_43665902.pdf)
* [Hao Yu, Bogdan M. Wilamowski Levenberg–Marquardt Training -- общий случай для плоских (персептрон) сетей](https://www.eng.auburn.edu/~wilambm/pap/2011/K10149_C012.pdf)


## Левенберг-Марквардт с кешированием в файловую систему

Позволяет снять ограничение с количества обучающих примеров. Кеш имеет достаточно большой размер, для оценки использовать:
$$M~\frac{\left[(L\times O) \times P + (L\times O)\right] \times 4}{2^{30}}$$
где $M$ - размер кеша в Gb; $L$ - количество обучающих объектов; $O$ - количество параметров сети; $P$ - количество выходных нейронов.

Например, для задачи с $L=100000$ обучающих объектов, сетью с $O=5000$ параметрами и $O=3$ выходными классами размер кеша составит `~1.2 Gb`,
знак не точного равенства, потому что не учитываются индексы и прочая служебная информация из `np.save`.

Эффективен в табличных задачах для построения бейзлайна, качество сопоставимо с XGBoost/CatBoost с поиском по сетке, однако,
сильно меньше параметров. **Не рекомендуется применять в задачах типа VAE с большим количеством выходных нейронов!**

Использование:

```python
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append("E:/reseach/LewenbergMarkvardt/lib")

from core.lm_distributed_samples import train_distributed_levenberg
from utils.visualisation_utils import plot_train_report
from utils.data_utils import TableLoader

# Батч устанавливаем так, чтобы хватило GPU, перемешивать данные не нужно - они все будут использованы за один шаг оптимизации
train_loader = DataLoader(TableLoader(x_train, y_train), batch_size=5000, shuffle=False)
val_loader = DataLoader(TableLoader(x_test, y_test), batch_size=5000, shuffle=False)

# Модель - может быть torch.nn.Sequential или torch.nn.Module -- ограничение только на количество параметров
# модель предварительно поместить на устройство вычислений
net = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], 200),
    torch.nn.Sigmoid(),
    torch.nn.Linear(200, y_train.shape[1]),
    torch.nn.Softmax(dim=1)
).cuda()

loss_history, val_history, mu_history, ep_time = train_distributed_levenberg(
    model,
    train_loader,
    torch.nn.MSELoss(),
    val_loader=val_loader,
    snap_folder=None,
    mu_init=10,
    min_error=1e-2,
    max_epochs=10,
    inner_steps=10,
    demping_coef=10,
    device="cuda:0",
    temp_folder="./temp",
)

# Сохраняем все метрики
pickle.dump(
    {"loss": loss_history, "val": val_history, "mu": mu_history, "time": ep_time},
    open("test_run.pkl", "wb"), 
)

# Визуализируем
plot_train_report("test_run.pkl")

# Объект model с настроенными весами - если не была указан `snap_folder`, нужно сохранить веса отдельно
torch.save(model.state_dict, 'weights.pkl')
```

**Описание аргументов:**
* `model (torch.nn.Module)`: модель, которая оптимизируется
* `x_loader (torch.utils.data.DataLoader)`: объект даталодера torch, batchsize будет соответсвовать размеру чанка
* `loss_fn (torch.nn.Module)`: функция ошибки
* `val_loader (Union[torch.utils.data.DataLoader, None])`: даталодер с валидационными данными (если есть - `min_error` счиатется на нем). `Defaults to None.`
* `snap_folder (Union[str, None] = None)`: папка в которую складывать модели, если нет - не будет чекпоинтов. `Defaults to None.`
* `mu_init (int, optional)`: начальное значение коэффициента регуляризации. `Defaults to 10.`
* `min_error (float, optional)`: минимальная ошибка при которой останавливается обучение. `Defaults to 1e-2.`
* `max_epochs (int, optional)`: максимльное число эпох при которых останавливается обучение. `Defaults to 10.`
* `inner_steps (int, optional)`: количество внутренних шагов. `Defaults to 10.`
* `demping_coef (float, optional)`: коэффициент изменения регуляризации. `Defaults to 10.`
* `device (str, optional)`: устройство вычислений. `Defaults to "cuda:0".`
* `temp_folder (str, optional)`: папка с кешами (удаляется после обучения). `Defaults to "./temp".`

**Возвращает:**
`Tuple[List[float], List[float], List[float], List[float]]`: кортеж со списками истории обучения: ошибка, ошибка валидации, регуляризатор (mu), время вычислений на эпоху
