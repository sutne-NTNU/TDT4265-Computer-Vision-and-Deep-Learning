from .utils import get_dataset_dir

# Import everything from the old dataset and only change the dataset folder.
from .baseline import (
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


data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir(
    "tdt4265_2022_updated/train_annotations.json"
)
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir(
    "tdt4265_2022_updated/val_annotations.json"
)
