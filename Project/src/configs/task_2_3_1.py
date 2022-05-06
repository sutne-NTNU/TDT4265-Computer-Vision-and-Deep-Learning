from tops.config import LazyCall as L
from ssd.modeling.backbones import FeaturePyramidNetwork

from .task_2_2 import (
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


backbone = L(FeaturePyramidNetwork)(
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
    backbone_out_channels=[256, 512, 1024, 2048, 512, 64],  # Using ResNet50
    outout_channels=128,
)
