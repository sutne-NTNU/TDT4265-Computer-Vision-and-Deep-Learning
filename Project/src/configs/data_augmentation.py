import torchvision

from tops.config import LazyCall as L
from ssd.data.transforms import ToTensor, GroundTruthBoxesToAnchors
from ssd.data.transforms import (
    ToTensor,
    RandomSampleCrop,
    RandomHorizontalFlip,
    ColorJitter,
    Resize,
    GroundTruthBoxesToAnchors,
)

from .baseline import (
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


train_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        # L(RandomSampleCrop)(),
        L(ToTensor)(),
        L(RandomHorizontalFlip)(),
        L(ColorJitter)(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.5),
        L(Resize)(imshape="${train.imshape}"),
        L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
    ]
)
