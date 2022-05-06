import torch
import numpy as np
from torch.nn import ModuleList, Sequential, Conv2d, ReLU
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


class SSD300(torch.nn.Module):
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        anchors,
        loss_objective,
        num_classes: int,
        use_deep_heads: bool = False,
        use_improved_weight_init: bool = False,
    ):
        super().__init__()
        """
        Implements the SSD network.
        Backbone outputs a list of features, which are gressed to SSD output with 
        regression/classification heads.
        """
        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.use_deep_heads = use_deep_heads
        self.use_improved_weight_init = use_improved_weight_init
        self.regression_heads = []
        self.classification_heads = []

        # Initialize output heads that are applied to each feature map from the backbone.
        for n_boxes, out_ch in zip(
            anchors.num_boxes_per_fmap,
            self.feature_extractor.out_channels,
        ):
            kernel_size, stride, padding = 3, 1, 1
            regr_out_ch = n_boxes * 4
            clsf_out_ch = n_boxes * self.num_classes

            if self.use_deep_heads:
                self.regression_heads.append(
                    Sequential(
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, regr_out_ch, kernel_size, stride, padding),
                    )
                )
                self.classification_heads.append(
                    Sequential(
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, out_ch, kernel_size, stride, padding),
                        ReLU(),
                        Conv2d(out_ch, clsf_out_ch, kernel_size, stride, padding),
                    )
                )

            else:
                self.regression_heads.append(
                    Conv2d(out_ch, regr_out_ch, kernel_size, stride, padding)
                )
                self.classification_heads.append(
                    Conv2d(out_ch, clsf_out_ch, kernel_size, stride, padding)
                )

        self.n_boxes_last = anchors.num_boxes_per_fmap[-1]
        self.regression_heads = ModuleList(self.regression_heads)
        self.classification_heads = ModuleList(self.classification_heads)
        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()

        self.regression_heads = ModuleList(self.regression_heads)
        self.classification_heads = ModuleList(self.classification_heads)
        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()

    def _init_weights(self):
        layers = [*self.regression_heads, *self.classification_heads]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        if self.use_improved_weight_init:
            p = 0.99
            bias = np.log(p * (self.num_classes - 1) / (1 - p))
            self.classification_heads[-1][-1].bias.data.fill_(bias)

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for idx, x in enumerate(features):
            bbox_delta = self.regression_heads[idx](x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads[idx](x).view(
                x.shape[0], self.num_classes, -1
            )
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    def forward(self, img: torch.Tensor, **kwargs):
        """
        img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)

    def forward_test(
        self,
        img: torch.Tensor,
        imshape=None,
        nms_iou_threshold=0.5,
        max_output=200,
        score_threshold=0.05,
    ):
        """
        img: shape: NCHW
        nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx],
                confs[img_idx],
                nms_iou_threshold,
                max_output,
                score_threshold,
            )
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions


def filter_predictions(
    boxes_ltrb: torch.Tensor,
    confs: torch.Tensor,
    nms_iou_threshold: float,
    max_output: int,
    score_threshold: float,
):
    """
    boxes_ltrb: shape [N, 4]
    confs: shape [N, num_classes]
    """
    assert 0 <= nms_iou_threshold <= 1
    assert max_output > 0
    assert 0 <= score_threshold <= 1
    scores, category = confs.max(dim=1)

    # 1. Remove low confidence boxes / background boxes
    mask = (scores > score_threshold).logical_and(category != 0)
    boxes_ltrb = boxes_ltrb[mask]
    scores = scores[mask]
    category = category[mask]

    # 2. Perform non-maximum-suppression
    keep_idx = batched_nms(
        boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold
    )

    # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
    keep_idx = keep_idx[:max_output]
    return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]
