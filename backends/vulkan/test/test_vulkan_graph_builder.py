# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.backends.vulkan.serialization.vulkan_graph_schema as schema
import torch
from executorch.backends.vulkan.serialization.vulkan_graph_builder import VkGraphBuilder
from executorch.backends.vulkan.vulkan_preprocess import apply_passes
from executorch.exir import to_edge
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.passes import SpecPropPass


def _build(module: torch.nn.Module, inputs) -> VkGraphBuilder:
    edge = to_edge(torch.export.export(module, inputs, strict=True))
    # The builder reads node specs, which the backend's own preprocess
    # populates before it gets here.
    program = apply_passes(edge.exported_program(), [SpecPropPass()])
    builder = VkGraphBuilder(
        program, DelegateMappingBuilder(generated_identifiers=True)
    )
    builder.build_graph()
    return builder


def _empty_builder() -> VkGraphBuilder:
    return VkGraphBuilder(None, DelegateMappingBuilder(generated_identifiers=True))


class TestVkGraphBuilderInputIds(unittest.TestCase):
    """The serialized input list has to match the delegate call's arguments.

    VulkanBackend::execute walks `args` positionally against
    ComputeGraph::inputs() and rejects the call when the counts disagree, so
    every placeholder that the delegate call passes must appear in input_ids,
    including ones this graph happens not to use. Unused placeholders are not
    hypothetical: passes that run after partitioning can fold away a
    placeholder's only consumers, leaving the argument list and the serialized
    graph out of step.
    """

    def test_unused_placeholder_is_still_declared_as_an_input(self) -> None:
        class UsesOnlyTheFirstInput(torch.nn.Module):
            def forward(self, used, unused):
                return used + used

        builder = _build(
            UsesOnlyTheFirstInput(), (torch.randn(2, 3), torch.randn(2, 3))
        )
        self.assertEqual(len(builder.input_ids), 2)

    def test_used_placeholders_are_declared_in_order(self) -> None:
        class UsesBothInputs(torch.nn.Module):
            def forward(self, first, second):
                return first + second

        builder = _build(UsesBothInputs(), (torch.randn(2, 3), torch.randn(2, 3)))
        self.assertEqual(len(builder.input_ids), 2)


class TestVkGraphBuilderScalarValues(unittest.TestCase):
    """Constant scalars are deduplicated by value; the key has to carry the
    type, because Python treats 256 and 256.0 (and 1 and True) as equal."""

    def test_grouped_conv_after_float_clamp_bound_gets_an_int_groups(self) -> None:
        # The original failure: a clamp bound of 256.0 cached a Double, and the
        # Int 256 that groups passes afterwards was handed that Double's id.
        class ClampThenDepthwiseConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(256, 256, 3, groups=256, bias=False)

            def forward(self, x):
                return self.conv(torch.nn.functional.hardtanh(x, -1.0, 256.0))

        builder = _build(ClampThenDepthwiseConv(), (torch.randn(1, 256, 4, 4),))
        conv = next(op for op in builder.chain if op.name == "aten.convolution.default")
        schema_args = [
            a.name for a in torch.ops.aten.convolution.default._schema.arguments
        ]
        groups = builder.values[conv.args[schema_args.index("groups")]].value
        self.assertIsInstance(groups, schema.Int)
        self.assertEqual(groups.int_val, 256)

    def test_equal_scalars_of_different_types_get_separate_values(self) -> None:
        builder = _empty_builder()

        double_id = builder.get_or_create_scalar_value(256.0)
        int_id = builder.get_or_create_scalar_value(256)
        bool_id = builder.get_or_create_scalar_value(True)
        one_id = builder.get_or_create_scalar_value(1)
        zero_id = builder.get_or_create_scalar_value(0.0)
        negative_zero_id = builder.get_or_create_scalar_value(-0.0)

        self.assertNotEqual(double_id, int_id)
        self.assertNotEqual(bool_id, one_id)
        self.assertNotEqual(zero_id, negative_zero_id)
        self.assertIsInstance(builder.values[double_id].value, schema.Double)
        self.assertIsInstance(builder.values[int_id].value, schema.Int)
        self.assertIsInstance(builder.values[bool_id].value, schema.Bool)
        self.assertIsInstance(builder.values[one_id].value, schema.Int)
        # and the cache still deduplicates equal scalars of the same type
        self.assertEqual(builder.get_or_create_scalar_value(256.0), double_id)
        self.assertEqual(builder.get_or_create_scalar_value(256), int_id)
        self.assertEqual(builder.get_or_create_scalar_value(-0.0), negative_zero_id)

    def test_int_cached_first_does_not_shadow_an_equal_double(self) -> None:
        builder = _empty_builder()

        int_id = builder.get_or_create_scalar_value(256)
        double_id = builder.get_or_create_scalar_value(256.0)

        self.assertNotEqual(int_id, double_id)
        self.assertIsInstance(builder.values[int_id].value, schema.Int)
        self.assertIsInstance(builder.values[double_id].value, schema.Double)


class TestVkGraphBuilderScalarLists(unittest.TestCase):
    """A scalar list serializes to exactly one Value. bool is a subclass of int,
    so a bool list also satisfies the all-int check, and an empty list
    satisfies every check. The builder is not told the argument's declared
    type, so an empty list is an IntList whatever the op's schema says; the
    empty-list test records that, it does not endorse it."""

    def setUp(self) -> None:
        self.builder = _empty_builder()

    def _values_appended_by(self, arg) -> list:
        before = len(self.builder.values)
        value_id = self.builder.create_scalar_list_value(arg)
        self.assertEqual(value_id, before)
        return [v.value for v in self.builder.values[before:]]

    def test_bool_list_serializes_once(self) -> None:
        values = self._values_appended_by([True, False])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], schema.BoolList)

    def test_empty_list_serializes_once(self) -> None:
        values = self._values_appended_by([])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], schema.IntList)

    def test_int_list_serializes_once(self) -> None:
        values = self._values_appended_by([1, 2])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], schema.IntList)

    def test_mixed_bool_and_int_list_serializes_as_ints(self) -> None:
        values = self._values_appended_by([True, 2])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], schema.IntList)
        self.assertEqual(values[0].items, [1, 2])
        self.assertTrue(all(type(e) is int for e in values[0].items))


if __name__ == "__main__":
    unittest.main()
