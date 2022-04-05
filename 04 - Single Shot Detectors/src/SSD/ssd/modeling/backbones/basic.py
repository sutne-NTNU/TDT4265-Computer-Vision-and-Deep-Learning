import torch
from typing import Tuple, List
from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """

    def __init__(self,
                 output_channels: List[int],
                 image_channels: int,
                 output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        kernel_size, padding = 3, 1
        self.layers = nn.ModuleList([

            nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size, 1, padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size, 1, padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size, 1, padding),
                nn.ReLU(),
                nn.Conv2d(64, output_channels[0], kernel_size, 2, padding),
                nn.ReLU(),
            ),  # 38x38

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[0], 128, kernel_size, 1, padding),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[1], kernel_size, 2, padding),
                nn.ReLU(),
            ),  # 19x19

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[1], 256, kernel_size, 1, padding),
                nn.ReLU(),
                nn.Conv2d(256, output_channels[2], kernel_size, 2, padding),
                nn.ReLU(),
            ),  # 9x9

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[2], 128, kernel_size, 1, padding),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[3], kernel_size, 2, padding),
                nn.ReLU(),
            ),  # 5x5

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[3], 128, kernel_size, 1, padding),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[4], kernel_size, 2, padding),
                nn.ReLU(),
            ),  # 3x3

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[4], 128, kernel_size, 1, padding),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[5], kernel_size, 1, 0),
                nn.ReLU(),
            )  # 1x1
        ])

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_feature = x

        # Perform Forward Pass
        for layer in self.layers:
            out_feature = layer(out_feature)
            out_features.append(out_feature)

        # Make sure output shapes are correct
        self.verify_out_features(out_features)

        # Return output features as tuple
        return tuple(out_features)

    def verify_out_features(self, out_features):
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
