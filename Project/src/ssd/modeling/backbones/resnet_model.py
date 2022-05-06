import torch
import torchvision
from torch.nn import ModuleList, Sequential, Conv2d, ReLU, BatchNorm2d
from torchvision.models.resnet import BasicBlock
from typing import List, Tuple


def custom_layer(in_channels, out_channels):
    return BasicBlock(
        inplanes=in_channels,
        planes=out_channels,
        stride=2,
        downsample=Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(out_channels),
        ),
    )


class ResnetModel(torch.nn.Module):
    """Wrapper class around a specific pretrained ResNet model to make sure shapes are correct"""

    def __init__(
        self,
        output_channels: List[int],
        image_channels: int,
        output_feature_sizes: List[Tuple[int]],
    ):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.layers = ModuleList(
            [
                # (3, 128, 1024)
                Sequential(
                    self.resnet.conv1,
                    self.resnet.bn1,
                    self.resnet.relu,
                    self.resnet.maxpool,
                    # (64, 32, 256)
                    self.resnet.layer1,  # output_channels[0]
                ),
                # (256, 32, 256)
                self.resnet.layer2,  # output_channels[1]
                # (512, 16, 128)
                self.resnet.layer3,  # output_channels[2]
                # (1024, 8, 64)
                self.resnet.layer4,  # output_channels[3]
                # (2048, 4, 32)
                custom_layer(self.out_channels[3], self.out_channels[4]),
                # (512, 2, 16)
                custom_layer(self.out_channels[4], self.out_channels[5]),
                # (64, 1, 8)
            ]
        )

    def forward(self, x):
        out_features = []

        for layer in self.layers:
            x = layer(x)
            out_features.append(x)

        self._verify_out_features(out_features)

        return tuple(out_features)

    def _verify_out_features(self, out_features):
        # Make sure output shapes are correct
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            shape = tuple(feature.shape[1:])
            assert (
                shape == expected_shape
            ), f"""ERROR at out_feature[{idx}]: 
                Expected shape: {expected_shape}
                           got: {shape}"""
        assert len(out_features) == len(
            self.output_feature_shape
        ), f"""ERROR, num outputted features was: 
            Expected num features: {len(out_features)}
            Actual num features:   {len(self.output_feature_shape)}"""
