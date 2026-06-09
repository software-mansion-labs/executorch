# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


# Edge IR targets we expect after `RMSNorm.use_rsqrt = True` decomposition.
_MUL_TARGETS = (
    exir_ops.edge.aten.mul.Tensor,
    torch.ops.aten.mul.Tensor,
)
_MEAN_TARGETS = (
    exir_ops.edge.aten.mean.dim,
    torch.ops.aten.mean.dim,
)
_ADD_TARGETS = (
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.add.Scalar,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
)
_RSQRT_TARGETS = (
    exir_ops.edge.aten.rsqrt.default,
    torch.ops.aten.rsqrt.default,
)
# Cast nodes are skipped when walking backward to the original input.
_CAST_TARGETS = (
    exir_ops.edge.aten._to_copy.default,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    exir_ops.edge.aten.to.dtype,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
)


def _is_call(node: Any, targets) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in targets
    )


def _skip_casts_backward(node: torch.fx.Node) -> torch.fx.Node:
    while isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target in _CAST_TARGETS:
        node = node.args[0]
    return node


def _is_eps(arg) -> bool:
    # eps is a small positive python scalar (or torch.fx.Node carrying one).
    if isinstance(arg, (float, int)):
        return True
    if isinstance(arg, torch.fx.Node):
        return arg.op != "call_function"
    return False


def _is_constant_weight(node) -> bool:
    """Return True iff `node` is a placeholder / get_attr that looks like a
    learned weight parameter (i.e. NOT the activation x)."""
    if not isinstance(node, torch.fx.Node):
        return False
    return node.op in ("get_attr", "placeholder")


def _verify_rms_inner(
    x_norm_node: torch.fx.Node,
) -> Optional[tuple]:
    """Verify `x_norm_node = mul(x_f32, rstd)` where rstd comes from the
    canonical (x*x → mean(-1) → +eps → rsqrt) chain.

    Returns (input_node, x_f32_node, eps_val, rstd_node, add_node, mean_node, sq_node)
    or None if not a match.
    """
    if not _is_call(x_norm_node, _MUL_TARGETS):
        return None
    xa, xb = x_norm_node.args[0], x_norm_node.args[1]
    rstd_node = None
    x_f32_node = None
    for cand_r, cand_x in ((xa, xb), (xb, xa)):
        if isinstance(cand_r, torch.fx.Node) and _is_call(cand_r, _RSQRT_TARGETS):
            rstd_node = cand_r
            x_f32_node = cand_x
            break
    if rstd_node is None or not isinstance(x_f32_node, torch.fx.Node):
        return None

    # plus_eps = add(mean_sq, eps)
    add_node = rstd_node.args[0]
    if not _is_call(add_node, _ADD_TARGETS):
        return None
    aa, ab = add_node.args[0], add_node.args[1]
    mean_node = None
    eps_val = None
    for cand_m, cand_e in ((aa, ab), (ab, aa)):
        if isinstance(cand_m, torch.fx.Node) and _is_call(cand_m, _MEAN_TARGETS):
            mean_node = cand_m
            eps_val = cand_e
            break
    if mean_node is None:
        return None
    if not _is_eps(eps_val):
        return None

    # mean_sq = mean(sq, dim=-1, keepdim=True)
    if len(mean_node.args) < 2:
        return None
    dims = mean_node.args[1]
    if not isinstance(dims, (list, tuple)) or len(dims) != 1:
        return None
    if dims[0] not in (-1,):
        return None

    # sq = mul(x_f32, x_f32)
    sq_node = mean_node.args[0]
    if not _is_call(sq_node, _MUL_TARGETS):
        return None
    sa, sb = sq_node.args[0], sq_node.args[1]
    if sa is not sb:
        return None
    if sa is not x_f32_node:
        return None

    # Walk back through any casts to find the original input x.
    input_node = _skip_casts_backward(x_f32_node)
    if not isinstance(input_node, torch.fx.Node):
        return None

    # eps as a python scalar.
    if isinstance(eps_val, torch.fx.Node):
        v = eps_val.meta.get("val", None)
        try:
            if hasattr(v, "item"):
                eps_val = float(v.item())
            elif v is not None:
                eps_val = float(v)
            else:
                return None
        except Exception:
            return None

    return (
        input_node,
        x_f32_node,
        float(eps_val),
        rstd_node,
        add_node,
        mean_node,
        sq_node,
    )


def _infer_normalized_shape(input_node: torch.fx.Node) -> Optional[list]:
    """Pull the hidden-size of the activation from its FakeTensor meta."""
    try:
        meta = input_node.meta.get("val")
        if meta is None:
            return None
        shape = list(meta.shape)
        if len(shape) == 0:
            return None
        return [int(shape[-1])]
    except Exception:
        return None


class RMSNormMatch(PatternMatch):
    """Match the Gemma4 RMSNorm (`use_rsqrt=True`) decomposed chain.

    Forward pattern (use_rsqrt=True, with_scale=True):

        x_f32        = to_dtype(x, fp32)              # optional cast
        sq           = mul(x_f32, x_f32)
        mean_sq      = mean(sq, dim=-1, keepdim=True)
        plus_eps     = add(mean_sq, eps)
        rstd         = rsqrt(plus_eps)
        x_norm       = mul(x_f32, rstd)
        x_scaled     = mul(x_norm, weight_f32)        # absent if with_scale=False
        out          = to_dtype(x_scaled, fp16)       # optional cast

    The anchor is the OUTERMOST mul (x_norm * weight) for with_scale=True.

    Captured nodes:
        - input_node:  original `x` (before any pre-cast)
        - weight_node: the weight tensor
        - eps:         python scalar
        - output_node: the final node whose users should be redirected
    """

    def __init__(self, anchor_mul: torch.fx.Node) -> None:
        self.anchor_node = anchor_mul
        self.match_found = False
        self.all_nodes = []
        self.input_node: Optional[torch.fx.Node] = None
        self.weight_node: Optional[torch.fx.Node] = None
        self.eps: float = 1e-6
        self.output_node: torch.fx.Node = anchor_mul
        self.normalized_shape: Optional[list] = None

        # anchor_mul: mul(x_norm, weight)
        if not _is_call(anchor_mul, _MUL_TARGETS):
            return
        a, b = anchor_mul.args[0], anchor_mul.args[1]

        # Decide which operand is the weight (constant-like placeholder).
        weight_node = None
        x_norm_node = None
        for cand_w, cand_x in ((a, b), (b, a)):
            cand_w_skipped = _skip_casts_backward(cand_w) if isinstance(cand_w, torch.fx.Node) else None
            if cand_w_skipped is not None and _is_constant_weight(cand_w_skipped):
                weight_node = cand_w_skipped
                x_norm_node = cand_x
                break
        if weight_node is None or not isinstance(x_norm_node, torch.fx.Node):
            return

        inner = _verify_rms_inner(x_norm_node)
        if inner is None:
            return
        (
            input_node,
            x_f32_node,
            eps_val,
            rstd_node,
            add_node,
            mean_node,
            sq_node,
        ) = inner

        # normalized_shape comes from the weight tensor's shape.
        try:
            wmeta = weight_node.meta.get("val")
            self.normalized_shape = list(wmeta.shape)
        except Exception:
            self.normalized_shape = None
        if self.normalized_shape is None or len(self.normalized_shape) == 0:
            return

        # The output of the chain is anchor_mul itself; the consumer might
        # immediately cast back to fp16 — leave that downstream cast in place
        # (the new op handles its own dtype via input dtype).
        self.input_node = input_node
        self.weight_node = weight_node
        self.eps = float(eps_val)
        self.output_node = anchor_mul

        # Track participating nodes (for later eliminate_dead_code visibility).
        self.all_nodes = [
            anchor_mul,
            x_norm_node,
            rstd_node,
            add_node,
            mean_node,
            sq_node,
        ]
        self.input_nodes = [self.input_node, self.weight_node]
        self.output_nodes = [self.output_node]
        self.match_found = True


class RMSNormNoWeightMatch(PatternMatch):
    """Match the Gemma4 RMSNorm (`use_rsqrt=True`, `with_scale=False`) chain.

    Forward pattern:
        x_f32        = to_dtype(x, fp32)              # optional cast
        sq           = mul(x_f32, x_f32)
        mean_sq      = mean(sq, dim=-1, keepdim=True)
        plus_eps     = add(mean_sq, eps)
        rstd         = rsqrt(plus_eps)
        x_norm       = mul(x_f32, rstd)               # ANCHOR
        out          = to_dtype(x_norm, fp16)         # optional cast

    The anchor is the `mul(x_f32, rstd)` itself (no trailing weight multiply).
    We MUST guard against double-matching with the with_scale=True case
    (where this mul is the inner step). We check that none of x_norm's
    consumers, after walking through casts, is a mul-by-weight that the
    `RMSNormMatch` detector would also accept.
    """

    def __init__(self, anchor_mul: torch.fx.Node) -> None:
        self.anchor_node = anchor_mul
        self.match_found = False
        self.all_nodes = []
        self.input_node: Optional[torch.fx.Node] = None
        self.weight_node: Optional[torch.fx.Node] = None
        self.eps: float = 1e-6
        self.output_node: torch.fx.Node = anchor_mul
        self.normalized_shape: Optional[list] = None

        if not _is_call(anchor_mul, _MUL_TARGETS):
            return

        inner = _verify_rms_inner(anchor_mul)
        if inner is None:
            return
        (
            input_node,
            x_f32_node,
            eps_val,
            rstd_node,
            add_node,
            mean_node,
            sq_node,
        ) = inner

        # Reject this anchor if any (cast-skipped) user is a mul-by-weight
        # — that's the with_scale=True path, handled by `RMSNormMatch`.
        # Walk forward through casts: anchor -> (maybe cast) -> consumer.
        def _is_with_scale_consumer(user: torch.fx.Node) -> bool:
            if not isinstance(user, torch.fx.Node):
                return False
            if user.op != "call_function":
                return False
            # If user is a cast, look further forward.
            if user.target in _CAST_TARGETS:
                return any(_is_with_scale_consumer(u) for u in user.users)
            # If user is a mul whose OTHER operand is a constant-like weight,
            # this is the with_scale=True chain.
            if user.target in _MUL_TARGETS:
                a, b = user.args[0], user.args[1]
                other = b if a is anchor_mul or (
                    isinstance(a, torch.fx.Node)
                    and a.op == "call_function"
                    and a.target in _CAST_TARGETS
                    and len(a.args) > 0
                    and a.args[0] is anchor_mul
                ) else a
                other_skipped = (
                    _skip_casts_backward(other)
                    if isinstance(other, torch.fx.Node)
                    else None
                )
                if other_skipped is not None and _is_constant_weight(other_skipped):
                    return True
            return False

        for user in list(anchor_mul.users):
            if _is_with_scale_consumer(user):
                return

        # No constant weight consumer — this is the with_scale=False path.
        normalized_shape = _infer_normalized_shape(input_node)
        if normalized_shape is None:
            return

        self.input_node = input_node
        self.weight_node = None
        self.eps = float(eps_val)
        self.output_node = anchor_mul
        self.normalized_shape = normalized_shape

        self.all_nodes = [
            anchor_mul,
            rstd_node,
            add_node,
            mean_node,
            sq_node,
        ]
        self.input_nodes = [self.input_node]
        self.output_nodes = [self.output_node]
        self.match_found = True


@register_pattern_detector("rms_norm")
def find_rms_norm_pattern(node: torch.fx.Node) -> Optional[RMSNormMatch]:
    # Anchor: the outermost mul `mul(x_norm, weight)`.
    if not _is_call(node, _MUL_TARGETS):
        return None
    m = RMSNormMatch(node)
    if not m.match_found:
        return None
    return m


@register_pattern_replacement("rms_norm")
def replace_rms_norm(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: RMSNormMatch,
) -> None:
    assert match.input_node is not None
    assert match.weight_node is not None
    assert match.normalized_shape is not None

    with graph_module.graph.inserting_before(match.anchor_node):
        new_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.rms_norm.default,
            args=(
                match.input_node,
                match.normalized_shape,
                match.weight_node,
                match.eps,
            ),
        )
    # The C++ rms_norm shader writes in the INPUT's dtype (fp16), not the
    # decomposed-chain anchor's dtype (fp32 after the to_dtype upcast). Tag
    # the replacement node with the input's meta so downstream `type_as(x)`
    # casts collapse to no-ops (eliminated by RemoveRedundantOpsTransform).
    if "val" in match.input_node.meta:
        new_node.meta["val"] = match.input_node.meta["val"]
    elif "val" in match.output_node.meta:
        new_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(new_node)


@register_pattern_detector("rms_norm_no_weight")
def find_rms_norm_no_weight_pattern(
    node: torch.fx.Node,
) -> Optional[RMSNormNoWeightMatch]:
    # Anchor: the inner `mul(x_f32, rstd)` whose result is NOT consumed by
    # a `mul(_, weight)` (those go through `RMSNormMatch`).
    if not _is_call(node, _MUL_TARGETS):
        return None
    m = RMSNormNoWeightMatch(node)
    if not m.match_found:
        return None
    return m


@register_pattern_replacement("rms_norm_no_weight")
def replace_rms_norm_no_weight(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: RMSNormNoWeightMatch,
) -> None:
    assert match.input_node is not None
    assert match.normalized_shape is not None

    with graph_module.graph.inserting_before(match.anchor_node):
        new_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.rms_norm.default,
            args=(
                match.input_node,
                match.normalized_shape,
                None,
                match.eps,
            ),
        )
    # See comment in `replace_rms_norm`: the shader writes in input dtype
    # (fp16), so the FX meta must track that to let the downstream cast
    # collapse.
    if "val" in match.input_node.meta:
        new_node.meta["val"] = match.input_node.meta["val"]
    elif "val" in match.output_node.meta:
        new_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(new_node)
