# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestFusePadIntoConvPass(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class PadConv(torch.nn.Module):
        """Asymmetric zero pad before a 4x4 filter, as in StyleGAN-style blur."""

        def __init__(self, value=0.0, depthwise=False):
            super().__init__()
            self.value = value
            self.conv = (
                torch.nn.Conv2d(8, 8, 4, groups=8, bias=False)
                if depthwise
                else torch.nn.Conv2d(8, 4, 4, padding=1)
            )

        def forward(self, x):
            x = torch.nn.functional.pad(x, (2, 1, 2, 1), value=self.value)
            return self.conv(x)

    class ConvPadConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(8, 4, 3)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
            return self.conv2(x)

    class PadConvTranspose(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(8, 4, 3)

        def forward(self, x):
            return self.conv(torch.nn.functional.pad(x, (1, 1, 1, 1)))

    def _run(self, module, inputs):
        (
            Tester(module.eval(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_pad_conv(self):
        self._run(self.PadConv(), (torch.randn(1, 8, 9, 7),))

    def test_fp32_pad_depthwise_conv(self):
        self._run(self.PadConv(depthwise=True), (torch.randn(1, 8, 9, 7),))

    def test_fp16_pad_conv(self):
        (
            Tester(self.PadConv().eval().half(), (torch.randn(1, 8, 9, 7).half(),))
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=2e-2, rtol=2e-2)
        )

    def test_fp32_conv_pad_conv(self):
        self._run(self.ConvPadConv(), (torch.randn(1, 3, 10, 12),))

    def test_fp32_nonzero_pad_value_not_fused_still_correct(self):
        self._run(self.PadConv(value=1.5), (torch.randn(1, 8, 9, 7),))

    def test_fp32_pad_conv_transpose_not_fused_still_correct(self):
        self._run(self.PadConvTranspose(), (torch.randn(1, 8, 5, 5),))
