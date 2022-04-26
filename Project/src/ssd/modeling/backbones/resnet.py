import torch
from torch import nn
from typing import Tuple, List


class FeaturePyramidNetwork(torch.nn.Module):
    def __init__(
        self,
        output_channels: List[int],
        image_channels: int,
        output_feature_sizes: List[Tuple[int]],
    ):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.layers = nn.ModuleList(
            [
                # Conv2d(output_channels, input_channels, kernel_size, stride, padding)
                nn.Sequential(
                    nn.Conv2d(image_channels, 32, 3, 1, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, output_channels[0], 4, 2, 1),
                    nn.ReLU(),
                ),  # 38x38
                nn.Sequential(
                    nn.Conv2d(output_channels[0], 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, output_channels[1], 4, 2, 1),
                    nn.ReLU(),
                ),  # 19x19
                nn.Sequential(
                    nn.Conv2d(output_channels[1], 256, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(256, output_channels[2], 4, 2, 1),
                    nn.ReLU(),
                ),  # 9x9
                nn.Sequential(
                    nn.Conv2d(output_channels[2], 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, output_channels[3], 4, 2, 1),
                    nn.ReLU(),
                ),  # 5x5
                nn.Sequential(
                    nn.Conv2d(output_channels[3], 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, output_channels[4], 4, 2, 1),
                    nn.ReLU(),
                ),  # 3x3
                nn.Sequential(
                    nn.Conv2d(output_channels[4], 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, output_channels[5], 4, 2, 1),
                    nn.ReLU(),
                ),  # 1x1
            ]
        )

    def forward(self, x):
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
            assert (
                feature.shape[1:] == expected_shape
            ), f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(
            self.output_feature_shape
        ), f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
