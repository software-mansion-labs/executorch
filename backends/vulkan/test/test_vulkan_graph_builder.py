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

    def test_float_and_int_of_equal_value_get_separate_values(self) -> None:
        import executorch.backends.vulkan.serialization.vulkan_graph_schema as schema

        builder = VkGraphBuilder.__new__(VkGraphBuilder)
        builder.values = []
        builder.const_scalar_to_value_ids = {}

        double_id = builder.get_or_create_scalar_value(256.0)
        int_id = builder.get_or_create_scalar_value(256)
        bool_id = builder.get_or_create_scalar_value(True)
        one_id = builder.get_or_create_scalar_value(1)

        self.assertNotEqual(double_id, int_id)
        self.assertNotEqual(bool_id, one_id)
        self.assertIsInstance(builder.values[double_id].value, schema.Double)
        self.assertIsInstance(builder.values[int_id].value, schema.Int)
        # and the cache still deduplicates equal scalars of the same type
        self.assertEqual(builder.get_or_create_scalar_value(256.0), double_id)
        self.assertEqual(builder.get_or_create_scalar_value(256), int_id)


class TestVkGraphBuilderScalarLists(unittest.TestCase):
    """A scalar list serializes to exactly one Value. bool is a subclass of int,
    so a bool list also satisfies the all-int check, and an empty list
    satisfies every check."""

    def setUp(self) -> None:
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x + x

        self.builder = _build(Identity(), (torch.randn(2),))

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


if __name__ == "__main__":
    unittest.main()
