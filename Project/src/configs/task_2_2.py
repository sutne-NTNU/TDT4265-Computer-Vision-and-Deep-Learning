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

# mAP gets really bad with any big augmentation, so keeping it low for now
train_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        # L(RandomSampleCrop)(),
        L(ToTensor)(),
        L(RandomHorizontalFlip)(p=0.3),
        # L(ColorJitter)(brightness=0.1, contrast=0, saturation=0.1, hue=0.1),
        L(Resize)(imshape="${train.imshape}"),
        L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
    ]
)
