import unittest
from unittest.mock import MagicMock

import torch
from executorch.backends.qualcomm import qnn_preprocess
from executorch.backends.qualcomm._passes import (
    AnnotateQuantAttrs,
    ConvertBmmToMatmul,
    ConvertMhaToSha,
    DecomposeMinMaxDim,
    FoldQDQ,
    InsertIOQDQ,
    InsertReshapeForReduceOps,
    RemoveRedundancy,
)
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnOperatorSupport
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.tests.models import (
    HardSigmoid,
    Reciprocal,
    TopKandIndex,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    qnn_edge_config,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
from executorch.exir.dialects._ops import ops as exir_ops
from torch.library import Library
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestPasses(unittest.TestCase):
    def _build_context_loader_edge_program(self, op_name, check_ir_validity=True):
        graph_name = "forward"
        custom_op = Library(OpContextLoader.namespace, "FRAGMENT")
        self.addCleanup(custom_op._destroy)
        custom_op.define(f"{op_name}(Tensor[] inputs) -> Any")

        @torch.library.impl(
            custom_op, op_name, dispatch_key="CompositeExplicitAutograd"
        )
        def op_impl(inputs):
            return (torch.zeros((1, 2), device="meta", dtype=inputs[0].dtype),)

        class Model(torch.nn.Module):
            def forward(self, x):
                return getattr(
                    getattr(torch.ops, OpContextLoader.namespace), op_name
                ).default((x,))

        exported_program = torch.export.export(
            Model(), (torch.ones(1, 2),), strict=True
        )
        compile_config = (
            None if check_ir_validity else EdgeCompileConfig(_check_ir_validity=False)
        )
        edge_program_manager = to_edge(
            {graph_name: exported_program},
            compile_config=compile_config,
        )
        edge_program = edge_program_manager._edge_programs[graph_name]
        context_loader_nodes = [
            node
            for node in edge_program.graph.nodes
            if node.op == "call_function"
            and OpContextLoader.namespace in str(node.target)
        ]
        return edge_program, context_loader_nodes

    def test_context_loader_edge_op_is_delegated(self):
        op_name = "ctx_loader_delegation"
        ctx_bin = b"qnn_context_binary"
        _, context_loader_nodes = self._build_context_loader_edge_program(
            op_name, check_ir_validity=False
        )
        self.assertEqual(1, len(context_loader_nodes))
        context_loader_nodes[0].meta[OpContextLoader.meta_ctx_bin] = ctx_bin

        # A fully constructed QnnOperatorSupport needs a live QNN manager, so
        # drive is_node_supported with a mock self: the context-loader node is
        # force-passed without touching instance state beyond the log label.
        support = MagicMock()
        support.phase = "QnnPartitioner"
        self.assertTrue(
            QnnOperatorSupport.is_node_supported(support, None, context_loader_nodes[0])
        )

    def test_build_op_wrappers_returns_context_binary(self):
        op_name = "ctx_loader_build"
        ctx_bin = b"qnn_context_binary"
        edge_program, context_loader_nodes = self._build_context_loader_edge_program(
            op_name, check_ir_validity=False
        )
        for node in context_loader_nodes:
            node.meta[OpContextLoader.meta_ctx_bin] = ctx_bin

        # For a graph whose only op is the context-binary loader, _build_op_wrappers
        # returns the stamped context binary directly, before any QNN compilation.
        result = QnnBackend._build_op_wrappers(
            edge_program,
            enable_tensor_dump=False,
            op_package_infos=[],
            use_mha2sha=False,
            backend_type=QnnExecuTorchBackendType.kHtpBackend,
        )
        self.assertEqual(ctx_bin, result)
        # Resolving the loader op must not leak torch names (e.g. range) into the module.
        self.assertNotIn("range", vars(qnn_preprocess))

    def test_context_loader_op_lowers_with_ir_validation(self):
        op_name = "ctx_loader_validation"

        # from_context_binary lowers with IR validity checks enabled (the
        # default). The context-loader custom op survives the edge verifier
        # because its namespace is not aten, so no validation is disabled and
        # the loader node is still present for downstream stamping.
        _, context_loader_nodes = self._build_context_loader_edge_program(
            op_name, check_ir_validity=True
        )
        self.assertEqual(1, len(context_loader_nodes))

    def _build_quantized_graph(self):
        """Build a quantized graph through AnnotateQuantAttrs + FoldQDQ."""

        class AddModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        module = AddModule().eval()
        sample_input = (torch.randn(1, 4),)

        exported = torch.export.export(module, sample_input, strict=True).module()
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_8a8w)
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*sample_input)
        qdq_module = convert_pt2e(prepared)

        edge_program = to_edge(
            torch.export.export(qdq_module, sample_input, strict=True)
        )
        ep = edge_program.exported_program()
        gm = ep.graph_module

        gm = AnnotateQuantAttrs(ep)(gm).graph_module
        gm = FoldQDQ(ep)(gm).graph_module
        return gm, ep

    def test_insert_io_qdq_handles_dequant_encoding(self):
        """InsertIOQDQ should not KeyError when a node with a dequantize
        encoding feeds the output node (e.g. pre-quantized LLM parameters)."""
        gm, ep = self._build_quantized_graph()

        # Wire b__frozen_param0 (which has dequantize encoding) to output,
        # simulating the LLM topology from github issue #17732.
        param_node = None
        output_node = None
        for n in gm.graph.nodes:
            if n.name == "b__frozen_param0":
                param_node = n
            if n.op == "output":
                output_node = n

        self.assertIsNotNone(param_node)
        old_args = output_node.args[0]
        output_node.args = (
            ((old_args,) if not isinstance(old_args, tuple) else old_args)
            + (param_node,),
        )
        gm.graph.lint()
        gm.recompile()

        pass_instance = InsertIOQDQ(ep)
        pass_instance._insert(gm)

        dq_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and hasattr(n.target, "__name__")
            and "dequantize" in n.target.__name__
            and any(u.op == "output" for u in n.users.keys())
        ]
        self.assertGreaterEqual(len(dq_nodes), 1)

    def test_insert_io_qdq_no_revisit(self):
        """InsertIOQDQ must not revisit newly inserted nodes."""
        gm, ep = self._build_quantized_graph()

        node_count_before = len(list(gm.graph.nodes))
        pass_instance = InsertIOQDQ(ep)
        pass_instance._insert(gm)
        node_count_after = len(list(gm.graph.nodes))

        # AddModule with one input and one output should insert exactly
        # one quantize (input) and one dequantize (output) = +2 nodes.
        self.assertEqual(node_count_after, node_count_before + 2)

    def test_decompose_minmaxdim_tensor_method(self):
        # Tensor.max carries no torch.max source_fn, which the pass used to need.
        class MaxMethod(torch.nn.Module):
            def forward(self, x):
                return x.max(dim=-1)

        x = torch.randn(2, 5, 7)
        ep = to_edge(torch.export.export(MaxMethod(), (x,))).exported_program()
        DecomposeMinMaxDim()(ep.graph_module)

        targets = [n.target for n in ep.graph.nodes]
        self.assertIn(exir_ops.edge.aten.argmax.default, targets)
        values, indices = ep.graph_module(x)
        ref_values, ref_indices = MaxMethod()(x)
        self.assertTrue(torch.equal(values, ref_values))
        self.assertTrue(torch.equal(indices, ref_indices))

    def test_insert_reshape_for_argmax(self):
        class ArgmaxModule(torch.nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=None)

        mod = ArgmaxModule()

        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        ep = torch.export.export(mod, (x,))
        # Run original module for reference
        ref = mod(x)

        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 0, "Reshape node not inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        InsertReshapeForReduceOps()(ep.graph_module)

        out = ep.graph_module(x)

        # Check graph structure: argmax should take a reshape as input
        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 1, "Reshape node should be inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        argmax_node = argmax_nodes[0]
        self.assertEqual(argmax_node.args[1], 0, "Argmax dim not set to 0")

        # Execute new graph and compare with reference
        out = ep.graph_module(x)
        self.assertTrue(
            torch.equal(*out, ref), f"Output mismatch: got {out}, expected {ref}"
        )

    def test_mha_to_sha(self):
        from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
        from executorch.examples.models.llama.model_args import ModelArgs
        from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
            CausalAttentionMask,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
            LlamaAttention,
        )

        # Initailize model config
        args = ModelArgs()
        args.max_seq_len = 128
        args.max_context_len = 128
        args.ar_len = 32
        args.use_kv_cache = True
        args.dim = 32
        args.n_heads = 8
        args.n_kv_heads = 8
        args.n_layers = 2
        args.head_dim = args.dim // args.n_heads
        mod = convert_linear_to_conv2d(LlamaAttention(0, args, True))

        # Prepare inputs
        hidden_states = torch.randn(args.max_batch_size, args.ar_len, args.dim)
        freqs_cos = torch.randn(args.ar_len, 1)
        freqs_sin = torch.randn(args.ar_len, 1)
        atten_mask = CausalAttentionMask(
            args.max_batch_size, args.ar_len, args.max_seq_len
        )
        k_cache = torch.zeros(
            args.max_batch_size,
            args.n_kv_heads,
            args.head_dim,
            args.max_seq_len - args.ar_len,
        )

        v_cache = torch.zeros(
            args.max_batch_size,
            args.n_kv_heads,
            args.max_seq_len - args.ar_len,
            args.head_dim,
        )
        sample_input = (
            hidden_states,
            freqs_cos,
            freqs_sin,
            atten_mask.mask,
            k_cache,
            v_cache,
        )

        # Run original module for reference
        refs = mod(*sample_input)

        # Export the module and convert linear to conv2d
        edge_program = to_edge(torch.export.export(mod, sample_input))
        new_ep = edge_program.exported_program()

        conv_nodes = [
            n
            for n in new_ep.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        # WQ, WK, WV, O
        self.assertTrue(len(conv_nodes) == 4, "Convolution nodes missing")

        # Convert MHA to SHA
        # This is a simplified version of what happens in the full pipeline to test the core functionality
        graph_module = RemoveRedundancy(quantization_capture=False)(
            new_ep.graph_module
        ).graph_module
        graph_module = ConvertBmmToMatmul()(graph_module).graph_module
        graph_module = ConvertMhaToSha(new_ep)(graph_module).graph_module

        conv_nodes = [
            n
            for n in new_ep.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        # Check graph structure: WQ, WK, WV should be converted to SHA
        self.assertTrue(len(conv_nodes) == 25, "Convolution nodes should be splited")

        # Execute new graph and compare with reference
        outs = graph_module(
            *new_ep.state_dict.values(), *new_ep.constants.values(), *sample_input
        )
        for i, (out, ref) in enumerate(zip(outs, refs)):
            self.assertTrue(
                torch.allclose(out, *ref, rtol=1e-6, atol=1e-6),
                f"Output {i} mismatch: got {out}, expected {ref}",
            )

    def test_resolve_debug_handle(self):
        name_handle_map = {
            "aten_topk_default": 1,
            "getitem": 1,
            "getitem_1": 1,
            "aten_view_copy_default": 2,
            "aten__to_copy_default": 3,
            "aten_index_tensor": 4,
            "aten_add_tensor": 5,
        }
        module = TopKandIndex()  # noqa: F405
        sample_input = (torch.randn(3, 10),)

        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,  # Random soc_model
            backend_options=backend_options,
            dump_intermediate_outputs=True,
        )

        try:
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module,
                sample_input,
                compiler_spec,
                generate_etrecord=True,
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise
        exec_prog_mgr = edge_prog_mgr.to_executorch()
        etrecord = exec_prog_mgr.get_etrecord()
        debug_handle_size = len(etrecord._debug_handle_map["forward"][0])
        self.assertEqual(
            len(name_handle_map),
            debug_handle_size,
            f"Number of handles does not match, expecting: {len(name_handle_map)}, but get: {debug_handle_size}",
        )
        after_edge_pass_ep = etrecord.graph_map["edge_after_transform/forward"]

        for node in after_edge_pass_ep.graph.nodes:
            if node.name in name_handle_map:
                expected_handle = name_handle_map.pop(node.name)
                node_handle = node.meta[DEBUG_HANDLE_KEY]
                self.assertEqual(
                    expected_handle,
                    node_handle,
                    f"{node.name} is expecting a handle id {expected_handle}, but got {node_handle}.",
                )
        self.assertEqual(
            len(name_handle_map),
            0,
            f"Following nodes did not find a match in the graph: {name_handle_map.keys()}",
        )

    def test_decompose_reciprocal_backend_aware(self):
        sample_input = (torch.tensor([2.0]),)
        target = torch.ops.aten.reciprocal.default
        decomposed_backends = (
            QnnExecuTorchBackendType.kHtpBackend,
            QnnExecuTorchBackendType.kGpuBackend,
            QnnExecuTorchBackendType.kLpaiBackend,
        )
        preserved_backends = (QnnExecuTorchBackendType.kUndefinedBackend,)

        for backend, should_decompose in [
            *[(b, True) for b in decomposed_backends],
            *[(b, False) for b in preserved_backends],
        ]:
            # The annotation pipeline is skipped for the GPU backend, as it does not support quantized data types
            pipelines = (
                ("export",)
                if backend == QnnExecuTorchBackendType.kGpuBackend
                else ("annotation", "export")
            )
            for pipeline in pipelines:
                with self.subTest(backend=backend, pipeline=pipeline):
                    ep = torch.export.export(Reciprocal(), sample_input, strict=True)
                    pm = get_qnn_pass_manager_cls(backend)()
                    if pipeline == "annotation":
                        pm.transform_for_annotation_pipeline(ep.graph_module)
                    else:
                        pm.transform_for_export_pipeline(ep)
                    has_target = any(
                        n.target == target for n in ep.graph_module.graph.nodes
                    )
                    self.assertNotEqual(
                        has_target,
                        should_decompose,
                        f"reciprocal {'should' if should_decompose else 'should NOT'} be decomposed for {backend.name}",
                    )

    def test_decompose_hardsigmoid_backend_aware(self):
        sample_input = (torch.tensor([2.0]),)
        target = torch.ops.aten.hardsigmoid.default
        decomposed_backends = (QnnExecuTorchBackendType.kLpaiBackend,)
        preserved_backends = (
            QnnExecuTorchBackendType.kGpuBackend,
            QnnExecuTorchBackendType.kHtpBackend,
            QnnExecuTorchBackendType.kUndefinedBackend,
        )

        for backend, should_decompose in [
            *[(b, True) for b in decomposed_backends],
            *[(b, False) for b in preserved_backends],
        ]:
            # The annotation pipeline is skipped for the GPU backend, as it does not support quantized data types
            pipelines = (
                ("export",)
                if backend == QnnExecuTorchBackendType.kGpuBackend
                else ("annotation", "export")
            )
            for pipeline in pipelines:
                with self.subTest(backend=backend, pipeline=pipeline):
                    ep = torch.export.export(HardSigmoid(), sample_input, strict=True)
                    pm = get_qnn_pass_manager_cls(backend)()
                    if pipeline == "annotation":
                        pm.transform_for_annotation_pipeline(ep.graph_module)
                    else:
                        pm.transform_for_export_pipeline(ep)
                    has_target = any(
                        n.target == target for n in ep.graph_module.graph.nodes
                    )
                    self.assertNotEqual(
                        has_target,
                        should_decompose,
                        f"hardsigmoid {'should' if should_decompose else 'should NOT'} be decomposed for {backend.name}",
                    )

    def test_partitioner_falls_back_on_op_without_visitor(self):
        """QnnOperatorSupport must reject an op that has no node visitor by returning
        False (CPU fallback), not by KeyError-ing on the node_visitors lookup.

        Uses aten.frac.default: it has no QNN node visitor and is not on any partition
        operator list, so it exercises the missing-visitor guard directly. (bitwise_not,
        the op that first surfaced this on Mamba2, is now in to_be_implemented_operator,
        which returns earlier and would not reach the guard.)
        """

        class FracModule(torch.nn.Module):
            def forward(self, x):
                return torch.frac(x) + x

        sample_input = (torch.randn(1, 4),)

        # Guard against a vacuous test: the visitor-less op must actually be present.
        exported = torch.export.export(FracModule().eval(), sample_input)
        self.assertTrue(
            any(
                node.op == "call_function" and "frac" in str(node.target)
                for node in exported.graph.nodes
            ),
            "expected aten.frac.default in the traced graph",
        )

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            # Must not raise KeyError: frac falls back to CPU; the rest may delegate.
            edge = to_edge_transform_and_lower_to_qnn(
                FracModule().eval(), sample_input, compiler_specs
            )
            edge.to_executorch()
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")

    def test_skipped_conv_keeps_its_weight_dequantize_on_cpu(self):
        """A quantized conv kept off QNN must take its per-channel weight dequantize
        with it. FoldQDQ force-passes that dequantize, so it used to be lowered alone
        as a QNN Dequantize, which rejects per-channel encodings and fails context
        binary generation (#19908)."""

        class ConvDepthwiseConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = torch.nn.Conv2d(8, 16, 1)
                self.dw = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16)
                self.pw2 = torch.nn.Conv2d(16, 8, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.dw(torch.relu(self.pw1(x)))))

        sample_input = (torch.randn(1, 8, 16, 16),)
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype=QuantDtype.use_16a8w, is_conv_per_channel=True
        )
        prepared = prepare_pt2e(
            torch.export.export(ConvDepthwiseConv().eval(), sample_input).module(),
            quantizer,
        )
        prepared(*sample_input)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=False),
        )
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                convert_pt2e(prepared),
                sample_input,
                compiler_specs,
                skip_node_id_set={"aten_convolution_default_1"},
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise

        # inspected before to_executorch(), which swaps in out-variant ops
        def cpu_nodes(op_name):
            return [
                node
                for node in edge.exported_program().graph.nodes
                if node.op == "call_function"
                and getattr(node.target, "__name__", "") == op_name
            ]

        convs = cpu_nodes("aten.convolution.default")
        self.assertEqual([conv.name for conv in convs], ["aten_convolution_default_1"])
        dequants = cpu_nodes("quantized_decomposed.dequantize_per_channel.default")
        self.assertTrue(dequants)
        for dequant in dequants:
            self.assertEqual(list(dequant.users), convs)
        edge.to_executorch()

    def test_block_size_is_per_node(self):
        """Every node in block_size_map must keep its own block size. The annotators
        used to write it into the weight observer constructor shared by all nodes, so
        the last node annotated set it for every one: a 2D linear block size reached a
        4D conv weight and calibration asserted in _get_reduction_params."""
        from executorch.backends.qualcomm.quantizer.observers.per_block_param_observer import (
            PerBlockParamObserver,
        )

        class ConvConvLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_a = torch.nn.Conv2d(32, 64, 1)
                self.conv_b = torch.nn.Conv2d(64, 64, 1)
                self.linear = torch.nn.Linear(64, 16)

            def forward(self, x):
                x = self.conv_b(torch.relu(self.conv_a(x)))
                return self.linear(x.mean(dim=(2, 3)))

        sample_input = (torch.randn(1, 32, 8, 8),)
        gm = torch.export.export(ConvConvLinear().eval(), sample_input).module()
        convs = [
            n.name for n in gm.graph.nodes if n.target == torch.ops.aten.conv2d.default
        ]
        linears = [
            n.name for n in gm.graph.nodes if n.target == torch.ops.aten.linear.default
        ]
        block_size_map = {
            convs[0]: (1, 16, 1, 1),
            convs[1]: (1, 32, 1, 1),
            linears[0]: (1, 16),
        }
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype=QuantDtype.use_16a4w_block,
            is_conv_per_channel=True,
            is_linear_per_channel=True,
        )
        quantizer.set_block_size_map(block_size_map)
        prepared = prepare_pt2e(gm, quantizer)
        for _ in range(2):
            prepared(*sample_input)

        observed = sorted(
            tuple(module.block_size)
            for module in prepared.modules()
            if isinstance(module, PerBlockParamObserver)
        )
        self.assertEqual(observed, sorted(block_size_map.values()))
        convert_pt2e(prepared)(*sample_input)

    def test_fp16a8w_conv_on_cpu_stays_fp16(self):
        """fp16a8w runs conv in fp16, but the HTP only takes it for 1x1 kernels, so a
        3x3 conv falls back to CPU. Its bias data and weight dequantize must be fp16
        as well, or to_executorch fails on a bias buffer that does not match its spec.
        """

        class ConvConvConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = torch.nn.Conv2d(8, 16, 1)
                self.k3 = torch.nn.Conv2d(16, 16, 3, padding=1)
                self.pw2 = torch.nn.Conv2d(16, 8, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.k3(torch.relu(self.pw1(x)))))

        sample_input = (torch.randn(1, 8, 16, 16),)
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype=QuantDtype.use_fp16a8w, is_conv_per_channel=True
        )
        prepared = prepare_pt2e(
            torch.export.export(ConvConvConv().eval(), sample_input).module(),
            quantizer,
        )
        prepared(*sample_input)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=False),
        )
        quantized = convert_pt2e(prepared)
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                quantized, sample_input, compiler_specs
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise

        program = edge.exported_program()
        convs = [
            node
            for node in program.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "__name__", "") == "aten.convolution.default"
        ]
        self.assertEqual([conv.name for conv in convs], ["aten_convolution_default_1"])
        weight, bias = convs[0].args[1], convs[0].args[2]
        self.assertEqual(weight.kwargs.get("out_dtype"), torch.float16)
        bias_name = program.graph_signature.inputs_to_parameters[bias.name]
        self.assertEqual(program.state_dict[bias_name].dtype, torch.float16)
        # the cast must not reach back into the caller's module
        self.assertEqual(
            {t.dtype for t in quantized.state_dict().values() if t.is_floating_point()},
            {torch.float32},
        )
        edge.to_executorch()

    def test_int_floor_divide_by_scalar_lowers(self):
        """An integer floor_divide by a Python scalar (YOLO26's `index // nc`) used to
        crash export.
        """

        class FloorDivideByScalar(torch.nn.Module):
            def forward(self, x):
                return x // 3

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                FloorDivideByScalar(),
                (torch.randint(-10, 10, (3, 4), dtype=torch.int32),),
                compiler_specs,
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise
        edge.to_executorch()

    def test_fuse_batch_norm_with_conv(self):
        """In HTP fp16 a standalone BatchNorm multiplies the convolution's rounding
        error by its per-channel scale; the to_edge pipeline folds it into the conv.
        A transposed conv keeps its BatchNorm: its weight is [in, out/groups, *k].
        conv1d is not covered: 1.4.1 rewrites it to conv2d between an unsqueeze and
        a squeeze before the edge passes, so its BatchNorm follows the squeeze.
        """

        class ConvBn(torch.nn.Module):
            def __init__(self, conv, bn):
                super().__init__()
                self.conv, self.bn = conv, bn

            def forward(self, x):
                return self.bn(self.conv(x))

        class DepthwiseConv2dBn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dw = torch.nn.Conv2d(8, 8, 3, groups=8, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(8)
                self.pw = torch.nn.Conv2d(8, 4, 1)
                self.bn2 = torch.nn.BatchNorm2d(4)

            def forward(self, x):
                x = torch.nn.functional.silu(self.bn1(self.dw(x)))
                return self.bn2(self.pw(x))

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        bn_ops = {
            exir_ops.edge.aten.native_batch_norm.default,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
        }
        conv = exir_ops.edge.aten.convolution.default

        def lower(module, shape):
            for m in module.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.running_mean.uniform_(-2, 2)
                    m.running_var.uniform_(1e-3, 1e-2)
            module.eval()
            ep = torch.export.export(module, (torch.randn(*shape),), strict=True)
            pass_manager = get_qnn_pass_manager_cls()()
            edge = to_edge_transform_and_lower(
                {"forward": pass_manager.transform_for_export_pipeline(ep)},
                transform_passes={
                    "forward": pass_manager.get_to_edge_transform_passes(
                        ep, compiler_specs=compiler_specs
                    )
                },
                partitioner=None,
                compile_config=qnn_edge_config(),
            )
            nodes = edge.exported_program().graph.nodes
            targets = [n.target for n in nodes if n.op == "call_function"]
            return sum(t in bn_ops for t in targets), targets.count(conv)

        cases = (
            ("depthwise_conv2d_bn", DepthwiseConv2dBn(), (1, 8, 16, 16), 2),
            (
                "conv3d_bn",
                ConvBn(torch.nn.Conv3d(4, 8, 3), torch.nn.BatchNorm3d(8)),
                (1, 4, 8, 8, 8),
                1,
            ),
        )
        for msg, module, shape, conv_count in cases:
            with self.subTest(msg=msg):
                self.assertEqual(lower(module, shape), (0, conv_count))
        with self.subTest(msg="conv_transpose2d_bn_is_skipped"):
            module = ConvBn(torch.nn.ConvTranspose2d(4, 8, 3), torch.nn.BatchNorm2d(8))
            self.assertEqual(lower(module, (1, 4, 8, 8)), (1, 1))
        # The folded weights are get_attr constants; partitioning must keep them
        # constants of the delegate, not turn them into runtime inputs.
        with self.subTest(msg="folded_weights_lower_to_qnn"):
            sample_input = (torch.randn(1, 8, 16, 16),)
            edge = to_edge_transform_and_lower_to_qnn(
                DepthwiseConv2dBn().eval(), sample_input, compiler_specs
            )
            edge.to_executorch()

    def test_unbind_of_quantized_constant_lowers(self):
        """AnnotateUnbind read the quant attrs from the quantize node feeding unbind's
        dequantize, but a folded quantized constant feeds the dequantize directly.
        """

        class UnbindConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

            def forward(self, x):
                a, b = torch.unbind(self.table, dim=0)
                return x * a + b

        sample_input = (torch.randn(4, 4),)
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_16a8w)
        prepared = prepare_pt2e(
            torch.export.export(UnbindConstant(), sample_input).module(), quantizer
        )
        prepared(*sample_input)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=False),
        )
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                convert_pt2e(prepared), sample_input, compiler_specs
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise
        edge.to_executorch()

    def test_constant_only_partition_stays_on_cpu(self):
        """An int64 buffer gets an int32 cast from I64toI32. When the buffer's consumer is
        not delegated, that cast alone would form a partition with no graph input, which
        QNN cannot compile ("No graph inputs present").
        """

        class Lookup(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.arange(10) % 3)

            def forward(self, x):
                values, indices = x.topk(4)
                return values + self.table[indices].float()

        sample_input = (torch.randn(10),)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                Lookup(),
                sample_input,
                compiler_specs,
                skip_node_op_set={"aten.index.Tensor", "aten.topk.default"},
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise

        edge.to_executorch()

    def test_full_with_runtime_fill_value_stays_on_cpu(self):
        """aten.full is lowered as a static tensor, so a fill_value only known at runtime
        (YOLO's `torch.full(..., stride.item())`) must leave it on CPU rather than crash
        the partitioner's support check.
        """

        class FullFromTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("stride", torch.tensor([8.0, 16.0]))

            def forward(self, x):
                return x * torch.full((4, 1), self.stride[0].item())

        sample_input = (torch.randn(4, 3),)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                FullFromTensor(), sample_input, compiler_specs
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise

        self.assertTrue(
            any(
                node.op == "call_function" and "aten.full" in str(node.target)
                for node in edge.exported_program().graph.nodes
            )
        )
        edge.to_executorch()


class TestIoBindingCheck(unittest.TestCase):
    """_check_io_binding must fire when QNN publishes I/O the signature lacks.

    The runtime binds delegate arguments positionally against the tensor list in
    the context binary, so a graph that declares more bindable tensors than the
    program passes reads past the end of args. That has to be caught at lowering.
    """

    @staticmethod
    def _edge_program(num_inputs, num_outputs):
        signature = MagicMock()
        signature.user_inputs = [f"arg_{i}" for i in range(num_inputs)]
        signature.user_outputs = [f"out_{i}" for i in range(num_outputs)]
        edge_program = MagicMock()
        edge_program.graph_signature = signature
        return edge_program

    def _run(self, names, num_inputs, num_outputs):
        from executorch.backends.qualcomm import qnn_preprocess

        wrappers = {f"n{i}": {0: object()} for i in range(len(names))}
        by_id = {
            id(next(iter(v.values()))): name
            for v, name in zip(wrappers.values(), names)
        }

        class _StubWrapper:
            def __init__(self, wrapper):
                self._name = by_id[id(wrapper)]

            def GetName(self):
                return self._name

        with unittest.mock.patch.object(
            qnn_preprocess.PyQnnManager, "PyQnnTensorWrapper", _StubWrapper
        ):
            qnn_preprocess._check_io_binding(
                self._edge_program(num_inputs, num_outputs), wrappers
            )

    def test_matching_io_passes(self):
        self._run(
            ["input_0_x", "output_y", "conv2d_3", "_frozen_param0"],
            num_inputs=1,
            num_outputs=1,
        )

    def test_mutable_buffers_do_not_consume_arguments(self):
        # A mutable buffer is threaded through separately and is not a user input.
        self._run(
            ["input_0_x", "input_1_mutbuf_0_cache", "output_y"],
            num_inputs=1,
            num_outputs=1,
        )

    def test_extra_published_inputs_are_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(
                ["input_0_x", "input_1_stray", "input_2_stray", "output_y"],
                num_inputs=1,
                num_outputs=1,
            )
        message = str(ctx.exception)
        self.assertIn("QNN declares 3 graph inputs", message)
        self.assertIn("input_1_stray", message)

    def test_extra_published_outputs_are_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(
                ["input_0_x", "output_y", "output_stray"], num_inputs=1, num_outputs=1
            )
        self.assertIn("2 graph outputs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
