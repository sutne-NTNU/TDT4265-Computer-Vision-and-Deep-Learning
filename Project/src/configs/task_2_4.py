import torchvision

from tops.config import LazyCall as L
from ssd.data.transforms import ToTensor, GroundTruthBoxesToAnchors, Resize
from ssd.modeling import SSD300, AnchorBoxes

from .task_2_3_4 import (
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

anchors = L(AnchorBoxes)(
    # unchanged
    image_shape="${train.imshape}",
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    scale_center_variance=0.1,
    scale_size_variance=0.2,
    # adjusted to fit dataset exploration results
    min_sizes=[
        [4, 4],
        [6, 6],
        [8, 8],
        [12, 12],
        [24, 24],
        [32, 32],
        [48, 48],
        [64, 64],
        [128, 128],
        [256, 256],
    ],
    aspect_ratios=[
        [4, 3, 2, 1],
        [4, 3, 2, 1],
        [4, 3, 2, 1],
        [5, 4, 3, 2, 1],
        [4, 3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
    ],
)
