import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn
from torch.nn import functional as F

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 2.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        The head will be applied to all levels of the feature pyramid, and predict 
        a single logit for each location on every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        """
        outputs = tuple()
        for feats in x:
            out = self.conv(feats)
            out = self.cls_logits(out)
            outputs += (out, )
        return outputs


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 2.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        The logic is rather similar to FCOSClassificationHead. The key difference is 
        that this head bundles both regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        """
        reg_outputs = tuple()
        ctr_outputs = tuple()
        for feats in x:
            out = self.conv(feats)
            reg_out = self.bbox_reg(out)
            ctr_out = self.bbox_ctrness(out)
            reg_outputs += (reg_out, )
            ctr_outputs += (ctr_out, )
        return reg_outputs, ctr_outputs


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FOCS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet18 is supported
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # fix all bn ops, yet still allow the update of affine params
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function changes depending on if the model is
    in training or evaluation mode.

    During training, the model expects both the input tensors
    (list of tensors within the range of [0, 1]), as well as a targets
    (list of dictionary), containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detection results during inference
            return detections

#########################################################################################
# Part II-2: FCOS training
#########################################################################################
    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """
    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        # Match the row-major order of each H x W point grid to the head outputs.
        cls_pred = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, self.num_classes)
             for p in cls_logits], dim=1
        )
        reg_pred = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, 4)
             for p in reg_outputs], dim=1
        )
        ctr_pred = torch.cat([p.flatten(1) for p in ctr_logits], dim=1)
        all_points = torch.cat([p.reshape(-1, 2) for p in points]).float()
        level_strides = torch.as_tensor(strides, device=all_points.device)
        level_ranges = torch.as_tensor(reg_range, device=all_points.device)
        point_strides = torch.cat([
            level_strides[i].expand(p.shape[0] * p.shape[1])
            for i, p in enumerate(points)
        ])
        point_ranges = torch.cat([
            level_ranges[i].expand(p.shape[0] * p.shape[1], 2)
            for i, p in enumerate(points)
        ])

        with torch.no_grad():
            batch_size, num_points = cls_pred.shape[:2]
            labels = torch.full(
                (batch_size, num_points), -1, dtype=torch.long,
                device=cls_pred.device,
            )
            reg_targets = all_points.new_zeros((batch_size, num_points, 4))
            # PointGenerator stores (y, x), whereas the boxes use (x1, y1, x2, y2).
            x = all_points[:, 1, None]
            y = all_points[:, 0, None]
            radius = self.center_sampling_radius * point_strides[:, None]
            for image_idx, target in enumerate(targets):
                boxes = target["boxes"].to(all_points)
                if boxes.numel() == 0:
                    continue
                distances = torch.stack((
                    x - boxes[None, :, 0], y - boxes[None, :, 1],
                    boxes[None, :, 2] - x, boxes[None, :, 3] - y,
                ), dim=-1)
                max_distance = distances.amax(dim=-1)
                eligible = (
                    (distances.amin(dim=-1) > 0)
                    & (max_distance >= point_ranges[:, None, 0])
                    & (max_distance <= point_ranges[:, None, 1])
                )
                if self.center_sampling_radius > 0:
                    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
                    eligible &= (
                        ((x - centers[None, :, 0]).abs() < radius)
                        & ((y - centers[None, :, 1]).abs() < radius)
                    )
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                # Resolve overlapping objects using the smallest eligible box.
                candidate_areas = areas[None, :].expand_as(eligible).masked_fill(
                    ~eligible, float("inf")
                )
                min_area, matched_idx = candidate_areas.min(dim=1)
                positive = min_area.isfinite()
                labels[image_idx, positive] = target["labels"][matched_idx[positive]]
                reg_targets[image_idx, positive] = (
                    distances[positive, matched_idx[positive]]
                    / point_strides[positive, None]
                )

            foreground = labels >= 0
            cls_targets = torch.zeros_like(cls_pred)
            image_idx, point_idx = foreground.nonzero(as_tuple=True)
            cls_targets[image_idx, point_idx, labels[foreground]] = 1
            positive_targets = reg_targets[foreground]
            left_right = positive_targets[:, [0, 2]]
            top_bottom = positive_targets[:, [1, 3]]
            ctr_targets = (
                (left_right.amin(dim=1) / left_right.amax(dim=1).clamp_min(1e-7))
                * (top_bottom.amin(dim=1) / top_bottom.amax(dim=1).clamp_min(1e-7))
            ).sqrt()
            if self.training:
                num_foreground = max(foreground.sum().item(), 1)
                self.loss_normalizer = (
                    self.loss_normalizer_momentum * self.loss_normalizer
                    + (1 - self.loss_normalizer_momentum) * num_foreground
                )

        normalizer = max(self.loss_normalizer, 1)
        cls_loss = sigmoid_focal_loss(
            cls_pred, cls_targets, reduction="sum"
        ) / normalizer
        # GIoU is translation/scale invariant. Decode relative to a common
        # origin in stride units to avoid duplicating absolute point coordinates.
        positive_pred = reg_pred[foreground].float()
        pred_boxes = torch.cat((-positive_pred[:, :2], positive_pred[:, 2:]), dim=-1)
        target_boxes = torch.cat(
            (-positive_targets[:, :2], positive_targets[:, 2:]), dim=-1
        )
        reg_loss = giou_loss(pred_boxes, target_boxes, reduction="sum") / normalizer
        ctr_loss = F.binary_cross_entropy_with_logits(
            ctr_pred[foreground].float(), ctr_targets, reduction="sum"
        ) / normalizer
        # Sum reduction also provides differentiable zero losses for empty GT.
        losses = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": cls_loss + reg_loss + ctr_loss,
        }
        return losses


#########################################################################################
# Part II-1: FCOS Inference
#########################################################################################
    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """
    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        detections = []
        for image_idx, (height, width) in enumerate(image_shapes):
            image_boxes, image_scores, image_labels = [], [], []
            for level, level_points in enumerate(points):
                cls_scores = cls_logits[level][image_idx].permute(1, 2, 0).reshape(
                    -1, self.num_classes
                ).float().sigmoid()
                ctr_scores = ctr_logits[level][image_idx].reshape(-1, 1).float().sigmoid()
                scores = (cls_scores * ctr_scores).sqrt().flatten()
                # Flatten point/category pairs so one point may predict several
                # classes; retain only the best candidates before decoding.
                candidate_idx = (scores > self.score_thresh).nonzero().flatten()
                scores = scores[candidate_idx]
                scores, order = scores.topk(min(self.topk_candidates, scores.numel()))
                candidate_idx = candidate_idx[order]
                point_idx = candidate_idx // self.num_classes
                labels = candidate_idx % self.num_classes
                distances = reg_outputs[level][image_idx].permute(1, 2, 0).reshape(
                    -1, 4
                )[point_idx].float()
                distances = distances * strides[level]
                selected_points = level_points.reshape(-1, 2)[point_idx]
                x, y = selected_points[:, 1], selected_points[:, 0]
                boxes = torch.stack((
                    x - distances[:, 0], y - distances[:, 1],
                    x + distances[:, 2], y + distances[:, 3],
                ), dim=-1)
                boxes[:, 0::2].clamp_(min=0, max=width)
                boxes[:, 1::2].clamp_(min=0, max=height)
                valid = (
                    (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                    & boxes.isfinite().all(dim=1)
                    & scores.isfinite()
                )
                image_boxes.append(boxes[valid])
                image_scores.append(scores[valid])
                image_labels.append(labels[valid])

            boxes = torch.cat(image_boxes)
            scores = torch.cat(image_scores)
            labels = torch.cat(image_labels)
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            detections.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep] + 1,
            })
        return detections
