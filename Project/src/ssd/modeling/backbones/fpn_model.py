from typing import OrderedDict, List, Tuple
import torch
import torchvision

from .resnet_model import ResnetModel


class FeaturePyramidNetwork(torch.nn.Module):
    """Feature Pyramid Network that uses my ResNet Wrapper model as backbone"""

    def __init__(
        self,
        outout_channels: int,
        backbone_out_channels: List[int],
        image_channels: int,
        output_feature_sizes: List[Tuple[int]],
    ):
        super().__init__()
        self.out_channels = [outout_channels] * 6

        self.backbone = ResnetModel(
            image_channels=image_channels,
            output_channels=backbone_out_channels,
            output_feature_sizes=output_feature_sizes,
        )

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=backbone_out_channels,
            out_channels=outout_channels,
        )

    def forward(self, x):
        features = self.backbone.forward(x)
        feature_map_dict = OrderedDict()
        for i, feature in enumerate(features):
            feature_map_dict[f"feat{i}"] = feature

        out_features = self.fpn(feature_map_dict).values()
        return tuple(out_features)
