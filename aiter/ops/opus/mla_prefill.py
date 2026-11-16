# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Raw ctypes bindings of the opus MLA kernels.

Each op is the C entry point of the same name in csrc/kernels/mla/opus_mla_prefill_fwd.cu (ported from aiter 541185a72, v4.1 gfx950 only).
ctypes binds positionally, so the argument order here must match the C signature one for one.
"""

import torch

from ...jit.core import compile_ops

PREFILL_MD_NAME = "module_mla_prefill_opus"


@compile_ops(PREFILL_MD_NAME, ffi_type="ctypes")
def opus_mla_v41_prefill_a8w8_gfx950_fwd(
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
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int = 1,
    partial_o: torch.Tensor | None = None,  # [N, num_splits, H, 512] fp32
    partial_max: torch.Tensor | None = None,  # [N, num_splits, H] fp32
    partial_sum: torch.Tensor | None = None,  # [N, num_splits, H] fp32
) -> None: ...


@compile_ops(PREFILL_MD_NAME, ffi_type="ctypes")
def opus_mla_v41_prefill_a8w4_gfx950_fwd(
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
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int = 1,
    partial_o: torch.Tensor | None = None,  # [N, num_splits, H, 512] fp32
    partial_max: torch.Tensor | None = None,  # [N, num_splits, H] fp32
    partial_sum: torch.Tensor | None = None,  # [N, num_splits, H] fp32
) -> None: ...
