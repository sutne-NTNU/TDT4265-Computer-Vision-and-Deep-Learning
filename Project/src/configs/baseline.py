import torchvision
import torch
from torch.optim.lr_scheduler import MultiStepLR, LinearLR

from tops.config import LazyCall as L
from ssd.modeling import SSD300, SSDMultiboxLoss, AnchorBoxes
from ssd.modeling.backbones import BaselineModel
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors
from ssd.data import TDT4265Dataset
from ssd.data.transforms import ToTensor, Normalize, Resize, GroundTruthBoxesToAnchors
from ssd.utils import batch_collate, batch_collate_val
from configs.utils import get_dataset_dir, get_output_dir


train = dict(
    batch_size=32,
    amp=True,  # Automatic mixed precision
    log_interval=20,
    seed=0,
    epochs=20,
    _output_dir=get_output_dir(),
    imshape=(128, 1024),
    image_channels=3,
)

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[
        [16, 16],
        [32, 32],
        [48, 48],
        [64, 64],
        [86, 86],
        [128, 128],
        [128, 400],
    ],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2,
)

backbone = L(BaselineModel)(
    output_channels=[128, 256, 128, 128, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
)

loss_objective = L(SSDMultiboxLoss)(anchors="${anchors}")

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background class
)

optimizer = L(torch.optim.SGD)(lr=5e-3, momentum=0.9, weight_decay=0.0005)

schedulers = dict(
    linear=L(LinearLR)(start_factor=0.1, end_factor=1, total_iters=500),
    multistep=L(MultiStepLR)(milestones=[], gamma=0.1),
)

train_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(ToTensor)(),
        L(Resize)(imshape="${train.imshape}"),
        L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
    ]
)

val_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(ToTensor)(),
        L(Resize)(imshape="${train.imshape}"),
    ]
)

gpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
    ]
)

data_train = dict(
    dataset=L(TDT4265Dataset)(
        img_folder=get_dataset_dir("tdt4265_2022"),
        transform="${train_cpu_transform}",
        annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"),
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}",
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        batch_size="${...train.batch_size}",
        collate_fn=batch_collate,
        drop_last=True,
    ),
    # GPU transforms can heavily speedup data augmentations.
    gpu_transform=gpu_transform,
)
data_val = dict(
    dataset=L(TDT4265Dataset)(
        img_folder=get_dataset_dir("tdt4265_2022"),
        transform="${val_cpu_transform}",
        annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json"),
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}",
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        batch_size="${...train.batch_size}",
        collate_fn=batch_collate_val,
    ),
    gpu_transform=gpu_transform,
)


label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
