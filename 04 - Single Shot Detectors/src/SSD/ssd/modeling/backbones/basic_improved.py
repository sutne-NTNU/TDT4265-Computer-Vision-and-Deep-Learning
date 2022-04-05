import torch
from typing import Tuple, List
from torch import nn

from .basic import BasicModel


class BasicImprovedModel(BasicModel):
    """
    Functions exactly the same as the BasicModel, but the layers are different
    """

    def __init__(self,
                 output_channels: List[int],
                 image_channels: int,
                 output_feature_sizes: List[Tuple[int]]):
        super().__init__(output_channels, image_channels, output_feature_sizes)

        self.layers = nn.ModuleList([

            nn.Sequential(
                nn.Conv2d(image_channels, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 5, 1, 2),
                nn.BatchNorm2d(128),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[0], 5, 2, 2),
                nn.ReLU(),
            ),  # 38x38

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[0], 256, 5, 1, 2),
                nn.ReLU(),
                nn.Conv2d(256, output_channels[1], 5, 2, 2),
                nn.ReLU(),
            ),  # 19x19

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[1], 512, 5, 1, 2),
                nn.ReLU(),
                nn.Conv2d(512, output_channels[2], 5, 2, 2),
                nn.ReLU(),
            ),  # 9x9

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[2], 256, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(256, output_channels[3], 3, 2, 1),
                nn.ReLU(),
            ),  # 5x5

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[3], 128, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[4], 3, 2, 1),
                nn.ReLU(),
            ),  # 3x3

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[4], 128, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[5], 3, 1, 0),
                nn.ReLU(),
            )  # 1x1
        ])
