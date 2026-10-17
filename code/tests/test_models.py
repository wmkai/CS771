"""CPU checks with synthetic inputs; no datasets or pretrained weights are needed.

Run from code/: python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cls_libs.modeling import Attention, TransformerBlock, default_cnn_model, SimpleViT
from det_libs.modeling import FCOS


def make_detector(num_classes=2):
    """Construct only the loss/postprocessing interface, without a backbone."""
    model = FCOS.__new__(FCOS)
    nn.Module.__init__(model)
    model.num_classes = num_classes
    model.center_sampling_radius = 1.5
    model.loss_normalizer = 1.0
    model.loss_normalizer_momentum = 0.0
    model.score_thresh = 0.1
    model.nms_thresh = 0.5
    model.topk_candidates = 100
    model.detections_per_img = 100
    return model


def feature(rows, requires_grad=False):
    """Convert batch x points x channels into NCHW for a one-row feature map."""
    return torch.tensor(rows, dtype=torch.float32).permute(0, 2, 1).unsqueeze(2).contiguous().requires_grad_(requires_grad)


def target(boxes=(), labels=()):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def focal_reference(logits, classes):
    """Scalar textbook focal loss with explicit class labels (-1 = background)."""
    total = 0.0
    for row, label in zip(logits, classes):
        for class_id, value in enumerate(row):
            probability = 1.0 / (1.0 + math.exp(-value))
            if class_id == label:
                total -= 0.25 * (1.0 - probability) ** 2 * math.log(probability)
            else:
                total -= 0.75 * probability ** 2 * math.log(1.0 - probability)
    return total / max(sum(label >= 0 for label in classes), 1)


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_attention_matches_independent_sdpa_on_rectangular_grid(self):
        attention = Attention(dim=12, num_heads=3).double()
        x = torch.randn(2, 2, 5, 12, dtype=torch.float64)
        # SDPA is independent of the manual batched-matmul implementation.
        q, k, v = F.linear(x, attention.qkv.weight, attention.qkv.bias).chunk(3, dim=-1)
        q, k, v = [t.reshape(2, 10, 3, 4).transpose(1, 2) for t in (q, k, v)]
        reference = F.scaled_dot_product_attention(q, k, v)
        reference = reference.transpose(1, 2).reshape(2, 2, 5, 12)
        reference = F.linear(reference, attention.proj.weight, attention.proj.bias)
        torch.testing.assert_close(attention(x), reference, atol=1e-10, rtol=1e-10)

    def test_zero_residual_branches_preserve_input_with_padded_windows(self):
        block = TransformerBlock(8, 2, window_size=3, drop_path=0.2)
        for layer in (block.attn.proj, block.mlp.fc2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        x = torch.randn(2, 4, 7, 8, requires_grad=True)
        result = block(x)
        torch.testing.assert_close(result, x, atol=0, rtol=0)
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)

    def test_window_partition_and_crop_with_independent_spatial_oracle(self):
        class WindowMean(nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2), keepdim=True).expand_as(x)

        class Zero(nn.Module):
            def forward(self, x):
                return x * 0

        block = TransformerBlock(2, 1, window_size=2)
        block.norm1 = block.norm2 = nn.Identity()
        block.attn = WindowMean()
        block.mlp = Zero()
        x = torch.arange(30, dtype=torch.float32).reshape(1, 3, 5, 2)
        expected = x.clone()
        for row in range(0, 3, 2):
            for col in range(0, 5, 2):
                patch = x[:, row:row + 2, col:col + 2]
                # Padding is zero, so every window still has four elements.
                expected[:, row:row + 2, col:col + 2] += patch.sum(dim=(1, 2), keepdim=True) / 4
        torch.testing.assert_close(block(x), expected)

    def test_cnn_and_vit_produce_logits_and_finite_gradients(self):
        models = [
            default_cnn_model(num_classes=7, widths=(8, 16, 32), blocks_per_stage=1),
            SimpleViT(img_size=32, patch_size=8, embed_dim=24, depth=2,
                      num_heads=3, num_classes=7, window_size=3,
                      window_block_indexes=(0,), drop_path_rate=0.1),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                images = torch.randn(2, 3, 40, 56, requires_grad=True)
                logits = model(images)
                self.assertEqual(logits.shape, (2, 7))
                self.assertTrue(torch.isfinite(logits).all())
                F.cross_entropy(logits, torch.tensor([1, 5])).backward()
                self.assertTrue(torch.isfinite(images.grad).all())
                self.assertGreater(images.grad.abs().sum().item(), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class DetectionInferenceTests(unittest.TestCase):
    def test_yx_stride_clipping_labels_and_multiple_images(self):
        model = make_detector()
        points = [torch.tensor([[[3.0, 9.0]]])]
        classes = [feature([[[math.log(4), -20]], [[-20, math.log(3)]]])]
        regression = [feature([[[1, 1, 2, 1]], [[2, 1, 1, 3]]])]
        centerness = [feature([[[math.log(9)]], [[math.log(4)]]])]
        result = model.inference(points, torch.tensor([2.0]), classes, regression,
                                 centerness, [(4, 10), (8, 16)])
        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0]["boxes"], torch.tensor([[7., 1., 10., 4.]]))
        torch.testing.assert_close(result[1]["boxes"], torch.tensor([[5., 1., 11., 8.]]))
        self.assertEqual(result[0]["labels"].tolist(), [1])
        self.assertEqual(result[1]["labels"].tolist(), [2])
        self.assertAlmostEqual(result[0]["scores"].item(), math.sqrt(0.8 * 0.9), places=6)
        self.assertAlmostEqual(result[1]["scores"].item(), math.sqrt(0.75 * 0.8), places=6)

    def test_nms_suppresses_same_class_but_keeps_other_class(self):
        model = make_detector()
        points = [torch.tensor([[[10., 20.], [10., 20.]]])]
        result = model.inference(
            points, torch.tensor([2.]),
            [feature([[[5., 4.], [3., 2.]]])],
            [feature([[[2., 2., 2., 2.], [2., 2., 2., 2.]]])],
            [feature([[[5.], [5.]]])], [(30, 40)],
        )[0]
        self.assertEqual(result["labels"].tolist(), [1, 2])
        torch.testing.assert_close(result["boxes"], torch.tensor([[16., 6., 24., 14.]]).repeat(2, 1))
        self.assertGreater(result["scores"][0], result["scores"][1])

    def test_topk_and_empty_images(self):
        model = make_detector(num_classes=1)
        model.topk_candidates = 1
        points = [torch.tensor([[[5., 5.], [5., 15.], [5., 25.]]])]
        result = model.inference(
            points, torch.tensor([1.]),
            [feature([[[1.], [2.], [4.]], [[-20.], [-20.], [-20.]]])],
            [feature([[[1., 1., 1., 1.]] * 3, [[1., 1., 1., 1.]] * 3])],
            [feature([[[8.]] * 3, [[8.]] * 3])], [(10, 30), (10, 30)],
        )
        torch.testing.assert_close(result[0]["boxes"], torch.tensor([[24., 4., 26., 6.]]))
        self.assertEqual(result[1]["boxes"].shape, (0, 4))
        self.assertEqual(result[1]["scores"].shape, (0,))
        self.assertEqual(result[1]["labels"].dtype, torch.int64)

    def test_box_outside_image_is_discarded_after_clipping(self):
        model = make_detector(num_classes=1)
        result = model.inference(
            [torch.tensor([[[10., 20.]]])], torch.tensor([1.]),
            [feature([[[9.]]])], [feature([[[1., 1., 1., 1.]]])],
            [feature([[[9.]]])], [(5, 5)],
        )[0]
        self.assertEqual(result["boxes"].shape, (0, 4))


class DetectionLossTests(unittest.TestCase):
    def assert_loss(self, actual, expected, places=5):
        self.assertTrue(torch.isfinite(actual).all())
        self.assertAlmostEqual(actual.item(), expected, places=places)

    def test_known_focal_giou_and_centerness_losses(self):
        model = make_detector()
        classes = [[math.log(4), -math.log(4)]]
        losses = model.compute_loss(
            [target([[4, 0, 20, 16]], [1])],
            [torch.tensor([[[8., 12.]]])], torch.tensor([4.]), torch.tensor([[0., 100.]]),
            [feature([classes], True)], [feature([[[1., 1., 1., 1.]]], True)],
            [feature([[[0.]]], True)],
        )
        self.assert_loss(losses["cls_loss"], focal_reference(classes, [1]))
        # Predicted box has one-quarter of target area and lies wholly inside it.
        self.assert_loss(losses["reg_loss"], 0.75)
        self.assert_loss(losses["ctr_loss"], math.log(2))
        self.assert_loss(losses["final_loss"], sum(losses[k].item() for k in ("cls_loss", "reg_loss", "ctr_loss")))

    def test_smallest_box_wins_independent_of_annotation_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                model = make_detector()
                boxes, labels = [[0, 0, 20, 20], [8, 6, 14, 14]], [0, 1]
                if reverse:
                    boxes, labels = boxes[::-1], labels[::-1]
                losses = model.compute_loss(
                    [target(boxes, labels)], [torch.tensor([[[10., 10.]]])],
                    torch.tensor([2.]), torch.tensor([[0., 100.]]),
                    [feature([[[2., -1.]]])], [feature([[[1., 2., 2., 2.]]])],
                    [feature([[[0.7]]])],
                )
                self.assert_loss(losses["cls_loss"], focal_reference([[2., -1.]], [1]))
                self.assert_loss(losses["reg_loss"], 0.)
                # Target centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)).
                self.assert_loss(losses["ctr_loss"], math.log1p(math.exp(0.7)) - 0.7 * math.sqrt(0.5))

    def test_center_sampling_excludes_interior_and_boundary_points(self):
        model = make_detector(num_classes=1)
        losses = model.compute_loss(
            [target([[0, 0, 20, 20]], [0])],
            [torch.tensor([[[5., 5.], [10., 10.], [7., 10.]]])],
            torch.tensor([2.]), torch.tensor([[0., 100.]]),
            [feature([[[0.], [0.], [0.]]])],
            [feature([[[1., 1., 1., 1.], [5., 5., 5., 5.], [1., 1., 1., 1.]]])],
            [feature([[[0.], [0.], [0.]]])],
        )
        self.assert_loss(losses["cls_loss"], focal_reference([[0.], [0.], [0.]], [-1, 0, -1]))
        self.assert_loss(losses["reg_loss"], 0.)
        self.assert_loss(losses["ctr_loss"], math.log(2))

    def test_regression_ranges_use_image_pixels_and_include_boundary(self):
        model = make_detector(num_classes=1)
        regression = [feature([[[1., 1., 1., 1.]]], True),
                      feature([[[2.5, 2.5, 2.5, 2.5]]], True)]
        losses = model.compute_loss(
            [target([[0, 0, 20, 20]], [0])],
            [torch.tensor([[[10., 10.]]]), torch.tensor([[[10., 10.]]])],
            torch.tensor([2., 4.]), torch.tensor([[0., 9.], [10., 20.]]),
            [feature([[[0.]]], True), feature([[[0.]]], True)], regression,
            [feature([[[0.]]], True), feature([[[0.]]], True)],
        )
        self.assert_loss(losses["cls_loss"], focal_reference([[0.], [0.]], [-1, 0]))
        self.assert_loss(losses["reg_loss"], 0.)
        losses["final_loss"].backward()
        torch.testing.assert_close(regression[0].grad, torch.zeros_like(regression[0]))

    def test_empty_ground_truth_has_finite_differentiable_losses(self):
        model = make_detector()
        classes = feature([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]], True)
        regression = feature([[[1., 1., 1., 1.]] * 2, [[1., 1., 1., 1.]] * 2], True)
        centerness = feature([[[0.], [0.]], [[0.], [0.]]], True)
        losses = model.compute_loss(
            [target(), target()], [torch.tensor([[[4., 8.], [4., 16.]]])],
            torch.tensor([4.]), torch.tensor([[0., 100.]]),
            [classes], [regression], [centerness],
        )
        self.assert_loss(losses["cls_loss"], focal_reference([[0., 0.]] * 4, [-1] * 4))
        self.assert_loss(losses["reg_loss"], 0.)
        self.assert_loss(losses["ctr_loss"], 0.)
        losses["final_loss"].backward()
        for tensor in (classes, regression, centerness):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(classes.grad.abs().sum().item(), 0)
        torch.testing.assert_close(regression.grad, torch.zeros_like(regression))
        torch.testing.assert_close(centerness.grad, torch.zeros_like(centerness))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
