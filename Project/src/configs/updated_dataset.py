from .utils import get_dataset_dir

# Import everything from the best model and only change the dataset folder.
# This file is also used for the dataset exploration.
from .task_2_4 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
)

images = get_dataset_dir("tdt4265_2022_updated")
train_annotations = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
val_annotations = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")

data_train.dataset.img_folder = images
data_train.dataset.annotation_file = train_annotations

data_val.dataset.img_folder = images
data_val.dataset.annotation_file = val_annotations
