import torch
import torchvision
import math
from .utils import get_output_dir
from ssd.data.transforms import Normalize
from ssd.modeling import backbones
from tops.config import LazyCall as L

# The line belows inherits the configuration set from ssd300, all changes to this are made below
from .ssd300 import (
    train,
    anchors,
    backbone,
    loss_objective,
    model,
    optimizer,
    schedulers,
    data_train,
    data_val,
    label_map
)

train.epochs = math.ceil(10_000/312)

# Backbone uses the improved model
backbone = L(backbones.BasicImprovedModel)(
    output_channels=[256, 512, 256, 128, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

optimizer = L(torch.optim.Adam)(
    lr=0.0005,
    weight_decay=0.0005
)

gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform
