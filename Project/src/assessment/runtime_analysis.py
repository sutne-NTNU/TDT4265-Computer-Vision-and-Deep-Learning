import time
import click
import torch
import tops
from ssd import utils
from pathlib import Path
from tops.config import instantiate
from tops.checkpointer import load_checkpoint


@torch.no_grad()
def evaluation(cfg, num_batches: int):
    model = instantiate(cfg.model)
    model.eval()
    model = tops.to_cuda(model)
    ckpt = load_checkpoint(
        cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device()
    )
    model.load_state_dict(ckpt["model"])
    dataloader_val = instantiate(cfg.data_val.dataloader)
    batch = next(iter(dataloader_val))
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    batch = tops.to_cuda(batch)
    batch = gpu_transform(batch)
    images = batch["image"]
    batch_size = int(images.shape[0])
    image_channels = int(images.shape[1])
    image_shape = tuple(images.shape[2:])

    # warmup
    for _ in range(10):
        model(images)
    # Measure Time
    start_time = time.time()
    for _ in range(num_batches):
        model(images)
    total_time = time.time() - start_time

    time_per_batch = total_time / num_batches
    bps = 1 / time_per_batch
    time_per_image = total_time / (batch_size * num_batches)
    fps = 1 / time_per_image

    print(
        f"""Runtime Analysis for images with shape={image_shape}, batch size={batch_size}: 
        Total Time={total_time:.2f} seconds ({num_batches} Batches)
        Batch Time={(1000*time_per_batch):.2f} milliseconds, BPS={bps:.2f}
        Image Time={(1000*time_per_image):.2f} milliseconds, FPS={fps:.2f}"""
    )


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("-n", "--n-images", default=100, type=int)
def main(config_path: Path, n_images: int):
    cfg = utils.load_config(config_path)
    evaluation(cfg, n_images)


if __name__ == "__main__":
    main()
