import torch

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
    # Unchanged
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2,
    # Changed to better match results of dataset exploration
    min_sizes=[
        [6, 6],
        [16, 16],
        [32, 32],
        [48, 48],
        [64, 64],
        [128, 128],
        [200, 200],
    ],
    aspect_ratios=[
        [2.0, 4.0],
        [1.3, 2.0],
        [1.3, 4.0],
        [1.5, 2.0, 4.0],
        [2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0],
    ],
)


optimizer = L(torch.optim.Adam)(
    lr=0.0005,
    weight_decay=0.0005,
)
