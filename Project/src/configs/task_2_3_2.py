from tops.config import LazyCall as L
from ssd.modeling import SSDMultiboxLoss

from .task_2_3_1 import (
    train,
    anchors,
    backbone,
    loss_objective,
    model,
    optimizer,
    schedulers,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    data_train,
    data_val,
    label_map,
)


loss_objective = L(SSDMultiboxLoss)(
    anchors="${anchors}",
    use_focal_loss=True,
)
