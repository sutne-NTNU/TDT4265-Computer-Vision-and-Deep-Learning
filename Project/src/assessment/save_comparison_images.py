import cv2
import os
import tops
import click
import numpy as np
from tops.config import instantiate
from tops.config import LazyCall as L
from tops.checkpointer import load_checkpoint
from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm
from ssd.data.transforms import ToTensor
from configs.utils import get_image_dir


def get_config(config_path):
    cfg = utils.load_config(config_path)
    cfg.train.batch_size = 1
    cfg.data_train.dataloader.shuffle = False
    cfg.data_val.dataloader.shuffle = False
    return cfg


def get_trained_model(cfg):
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(
        cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device()
    )
    model.load_state_dict(ckpt["model"])
    return model


def get_dataloader(cfg, dataset_to_visualize):
    # We use just to_tensor to get rid of all data augmentation, etc...
    to_tensor_transform = [L(ToTensor)()]
    if dataset_to_visualize == "train":
        cfg.data_train.dataset.transform.transforms = to_tensor_transform
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataset.transform.transforms = to_tensor_transform
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def convert_boxes_coords_to_pixel_coords(boxes, width, height):
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes.cpu().numpy()


def convert_image_to_hwc_byte(image):
    first_image_in_batch = image[0]  # This is the only image in batch
    image_pixel_values = (first_image_in_batch * 255).byte()
    image_h_w_c_format = image_pixel_values.permute(1, 2, 0)
    return image_h_w_c_format.cpu().numpy()


def visualize_annotations_on_image(image, batch, label_map):
    boxes = convert_boxes_coords_to_pixel_coords(
        batch["boxes"][0], batch["width"], batch["height"]
    )
    labels = batch["labels"][0].cpu().numpy().tolist()

    image_with_boxes = draw_boxes(image, boxes, labels, class_name_map=label_map)
    return image_with_boxes


def visualize_model_predictions_on_image(
    image, img_transform, batch, model, label_map, score_threshold
):
    pred_image = tops.to_cuda(batch["image"])
    transformed_image = img_transform({"image": pred_image})["image"]

    boxes, categories, scores = model(
        transformed_image, score_threshold=score_threshold
    )[0]
    boxes = convert_boxes_coords_to_pixel_coords(
        boxes.detach().cpu(), batch["width"], batch["height"]
    )
    categories = categories.cpu().numpy().tolist()

    image_with_predicted_boxes = draw_boxes(
        image, boxes, categories, scores, class_name_map=label_map
    )
    return image_with_predicted_boxes


def create_filepath(save_folder, image_id):
    filename = "image_" + str(image_id) + ".png"
    return os.path.join(save_folder, filename)


def create_comparison_image(batch, model, img_transform, label_map, score_threshold):
    image = convert_image_to_hwc_byte(batch["image"])
    image_with_annotations = visualize_annotations_on_image(image, batch, label_map)
    image_with_model_predictions = visualize_model_predictions_on_image(
        image, img_transform, batch, model, label_map, score_threshold
    )

    concatinated_image = np.concatenate(
        [image, image_with_annotations, image_with_model_predictions], axis=0
    )
    return concatinated_image


def create_and_save_comparison_images(dataloader, model, cfg, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Saving images to", save_folder)
    num_skips = 25  # only use every 25th image
    num_images_to_save = len(dataloader) // num_skips
    dataloader = iter(dataloader)

    img_transform = instantiate(cfg.data_val.gpu_transform)
    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        for _ in range(num_skips - 1):
            next(dataloader)
        comparison_image = create_comparison_image(
            batch, model, img_transform, cfg.label_map, 0.3
        )
        filepath = create_filepath(save_folder, i)
        cv2.imwrite(filepath, comparison_image[:, :, ::-1])


def get_save_folder_name(cfg):
    return os.path.join(get_image_dir(), "comparison_images", cfg.run_name)


@click.command()
@click.argument("config_path")
def main(config_path):
    cfg = get_config(config_path)
    model = get_trained_model(cfg)

    dataloader = get_dataloader(cfg, "val")
    save_folder = get_save_folder_name(cfg)

    create_and_save_comparison_images(dataloader, model, cfg, save_folder)


if __name__ == "__main__":
    main()
