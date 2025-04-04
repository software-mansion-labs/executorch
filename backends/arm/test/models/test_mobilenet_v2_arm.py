# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from torchvision import models, transforms
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestMobileNetV2(unittest.TestCase):
    """Tests MobileNetV2."""

    mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    mv2 = mv2.eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    model_inputs = (normalize(torch.randn((1, 3, 224, 224))),)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
    }

    operators_after_quantization = all_operators - {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
    }

    def test_mv2_tosa_MI(self):
        (
            ArmTester(
                self.mv2,
                example_inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80.0+MI", permute_memory_to_nhwc=True
                ),
            )
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=self.model_inputs)
        )

    def test_mv2_tosa_BI(self):
        (
            ArmTester(
                self.mv2,
                example_inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80.0+BI", permute_memory_to_nhwc=True
                ),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            # atol=1.0 is a defensive upper limit
            # TODO MLETROCH-72
            # TODO MLETROCH-149
            .run_method_and_compare_outputs(atol=1.0, qtol=1, inputs=self.model_inputs)
        )

    def test_mv2_u55_BI(self):
        tester = (
            ArmTester(
                self.mv2,
                example_inputs=self.model_inputs,
                compile_spec=common.get_u55_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0, qtol=1, inputs=self.model_inputs, target_board="corstone-300"
            )

    def test_mv2_u85_BI(self):
        tester = (
            ArmTester(
                self.mv2,
                example_inputs=self.model_inputs,
                compile_spec=common.get_u85_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0, qtol=1, inputs=self.model_inputs, target_board="corstone-320"
            )
