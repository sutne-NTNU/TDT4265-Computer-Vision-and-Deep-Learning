from .utils import get_dataset_dir

# Import everything from the best model and only change the dataset
# This file is also used for the dataset exploration.
from .task_2_4 import (
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

images = get_dataset_dir("tdt4265_2022_updated")
train_annotations = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
val_annotations = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")

data_train.dataset.img_folder = images
data_train.dataset.annotation_file = train_annotations

data_val.dataset.img_folder = images
data_val.dataset.annotation_file = val_annotations
