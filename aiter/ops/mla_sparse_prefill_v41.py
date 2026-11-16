# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA v4.1 sparse prefill on gfx950 (opus), ported from aiter 541185a72
``aiter/ops/mla_sparse_prefill.py`` (``mla_sparse_prefill_v41``, #6675 / #7124): one
softmax per query token over its rows of a prefix source (``unified_kv``) and of an
extend source (``kv``), each addressed by int32 CSR row lists, with a per-head sink in
the denominator only. Rows are 512 values with separate per-32 E8M0 scales (q, kv fp8);
the prefix source is fp8 or NVFP4 (e2m1 pairs + per-16 e4m3 scales)."""

import functools
import math
from typing import NamedTuple

import torch

from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..utility import dtypes
from .opus.mla_prefill import (
    opus_mla_v41_prefill_a8w4_gfx950_fwd,
    opus_mla_v41_prefill_a8w8_gfx950_fwd,
)

__all__ = ["mla_sparse_prefill_v41"]


def _out_bf16(t: int, h: int, device, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return torch.empty((t, h, 512), dtype=torch.bfloat16, device=device)
    if out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )
    return out


# Auto split cost in keys: one workgroup's walk (a token's keys over S splits, in whole KV tiles),
# plus c0 + c1 * N * H * S to fold when split.
class _SplitFit(NamedTuple):
    heads_per_wg: int
    kv_tile: int
    wgs_per_cu: int  # workgroups per CU a split may fill
    counts: tuple  # the split counts auto picks from
    c0: float
    c1: float
    packed: float = 1.0  # walk factor past one workgroup per CU


# The reduce gives each split a lane of one wave, so a wave32 arch caps the count.
_MAX_KV_SPLITS = 32
_ANY_COUNT = tuple(range(2, _MAX_KV_SPLITS + 1))

# kernel -> fit over H 16-128, topk 384-4096 and N 8-768 on gfx950.
_KV_SPLIT_FIT = {
    "v41_a8w8_16mx1_16nx4": _SplitFit(16, 64, 2, _ANY_COUNT, 80, 0.046),
    "v41_a8w8_16mx4_64nx1": _SplitFit(64, 64, 1, _ANY_COUNT, 90, 0.025),
    "v41_a8w4_16mx1_16nx4": _SplitFit(16, 64, 2, _ANY_COUNT, 80, 0.046),
    "v41_a8w4_16mx4_64nx1": _SplitFit(64, 64, 1, _ANY_COUNT, 250, 0.012),
}


def _opus_split_fit(kernel: str, h: int) -> _SplitFit:
    tile = "16mx1_16nx4" if h <= 16 else "16mx4_64nx1"
    return _KV_SPLIT_FIT[f"{kernel}_{tile}"]


@functools.lru_cache(maxsize=1024)
def _auto_kv_splits(fit: _SplitFit, n: int, h: int, nnz: int, cu_num: int) -> int:
    if n == 0:
        return 1
    keys = nnz / n
    blocks = n * -(-h // fit.heads_per_wg)
    max_splits = fit.wgs_per_cu * cu_num // blocks

    def cost(s: int) -> float:
        walk = -(-math.ceil(keys / s) // fit.kv_tile) * fit.kv_tile
        if s > 1 and blocks * s > cu_num:
            walk *= fit.packed
        return walk + (fit.c0 + fit.c1 * n * h * s if s > 1 else 0)

    return min([1, *(s for s in fit.counts if s <= max_splits)], key=cost)


def _kv_splits(
    num_kv_splits: int | None, fit: _SplitFit, n: int, h: int, nnz: int
) -> int:
    """The split count. Auto reads host scalars only (``nnz``: all tokens' rows, from the
    index tensors' sizes), so a captured graph's choice is its shape's."""
    if num_kv_splits is None:
        return _auto_kv_splits(fit, n, h, nnz, get_cu_num())
    if not 1 <= num_kv_splits <= _MAX_KV_SPLITS:
        raise ValueError(
            f"num_kv_splits must be in [1, {_MAX_KV_SPLITS}], got {num_kv_splits}"
        )
    return num_kv_splits


def _split_args(n: int, num_kv_splits: int, h: int, device) -> tuple:
    """The opus kernels' trailing split-K arguments, none without split-K: num_splits,
    partial_o [N, S, H, 512], partial_max and partial_sum [N, S, H], fp32."""
    if num_kv_splits == 1:
        return ()
    partial_o = torch.empty(
        n, num_kv_splits, h, 512, dtype=torch.float32, device=device
    )
    partial_max = torch.empty(n, num_kv_splits, h, dtype=torch.float32, device=device)
    return num_kv_splits, partial_o, partial_max, torch.empty_like(partial_max)


_E8M0 = (torch.uint8, dtypes.fp8_e8m0)

# unified_kv dtype -> (kernel, split-fit variant, unified_kv_scale dtypes); q and kv are fp8.
_OPUS_V41_KERNELS = {
    dtypes.fp8: (opus_mla_v41_prefill_a8w8_gfx950_fwd, "v41_a8w8", _E8M0),
    dtypes.fp4x2: (
        opus_mla_v41_prefill_a8w4_gfx950_fwd,
        "v41_a8w4",
        (torch.uint8, dtypes.fp8),
    ),
}


def mla_sparse_prefill_v41(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
) -> torch.Tensor:
    """Args (as 541185a72 ``mla_sparse_prefill_v41``):
      q:                 ``[T, H, 512]`` fp8 query; q_scale ``[T, H, 16]`` E8M0.
      unified_kv:        ``[rows, 512]`` fp8 prefix rows (or ``[rows, 256]`` NVFP4) with
        unified_kv_scale ``[rows, 16]`` E8M0 (or ``[rows, 32]`` e4m3); row strides free.
      kv_indices_prefix / kv_indptr_prefix: int32 CSR row lists into the prefix source.
      kv / kv_scale:     ``[rows, 512]`` fp8 extend rows, ``[rows, 16]`` E8M0.
      kv_indices_extend / kv_indptr_extend: int32 CSR row lists into the extend source.
      attn_sink:         ``[H]`` fp32 or None.
    Returns ``out`` (``[T, H, 512]`` bf16)."""
    if q.dtype != dtypes.fp8 or kv.dtype != dtypes.fp8:
        raise RuntimeError(f"q and kv must be fp8, got q={q.dtype}, kv={kv.dtype}")
    prefix_dtype = dtypes.fp4x2 if unified_kv.dtype == torch.uint8 else unified_kv.dtype
    if prefix_dtype not in _OPUS_V41_KERNELS:
        raise RuntimeError(
            f"mla_sparse_prefill_v41 supports unified_kv dtypes "
            f"{list(_OPUS_V41_KERNELS)}, got {unified_kv.dtype}"
        )
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"mla_sparse_prefill_v41 (ported) supports gfx950, got {gfx}")
    fwd, variant, prefix_scale_dtypes = _OPUS_V41_KERNELS[prefix_dtype]
    for name, scale, scale_dtypes in (
        ("q_scale", q_scale, _E8M0),
        ("unified_kv_scale", unified_kv_scale, prefix_scale_dtypes),
        ("kv_scale", kv_scale, _E8M0),
    ):
        if scale.dtype not in scale_dtypes:
            raise RuntimeError(
                f"{name} must be one of {scale_dtypes}, got {scale.dtype}"
            )
    n, h, _ = q.shape
    out = _out_bf16(n, h, q.device, out)
    nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
    fit = _opus_split_fit(variant, h)
    num_kv_splits = _kv_splits(num_kv_splits, fit, n, h, nnz)
    fwd(
        q,
        q_scale,
        unified_kv,
        unified_kv_scale,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_scale,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
        *_split_args(n, num_kv_splits, h, q.device),
    )
    return out
