"""
Unit test for mhc_pre_gemm_sqrsum correctness, specifically targeting
the sqrsum store race condition in mhc_kernels.cu.

Bug: In mhc_pre_gemm_sqrsum_kernel, the sqrsum store uses raw `lane_id`
instead of `lane_id % mfma_m` for the row index. Since mfma_m=16 and
warp_size=64, lanes 16-63 within each warp write their (reduced) sqrsum
to indices that belong to OTHER warps' rows, causing a race condition.

The race corrupts sqrsum values for rows >= 16 within each tile_m block,
which cascades into incorrect RMS normalization -> incorrect mixes ->
incorrect pre/post/comb -> lower model accuracy.

To reproduce reliably, we scale each row differently so that the sqrsum
values are distinct per row. If the race replaces row N's sqrsum with
row (N % 16)'s sqrsum, the error is clearly detectable.
"""

import torch
import unittest


def torch_ref_gemm_sqrsum(x_bf16, fn_fp32):
    """Pure torch reference: GEMM + sum-of-squares."""
    x_fp32 = x_bf16.float()
    sqrsum = x_fp32.square().sum(dim=-1)
    out = x_fp32 @ fn_fp32.T
    return out, sqrsum


def torch_ref_mhc_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    """Pure torch reference for the full mhc_pre pipeline."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    x_flat = residual.flatten(-2, -1).float()
    sqrsum = x_flat.square().sum(-1)
    out = x_flat @ fn.T
    rsqrt_val = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    mixes = out * rsqrt_val.unsqueeze(-1)

    expanded_scale = torch.cat([
        hc_scale[0].expand(hc_mult),
        hc_scale[1].expand(hc_mult),
        hc_scale[2].expand(hc_mult * hc_mult),
    ])
    mixes = mixes * expanded_scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)

    x = res_mix.softmax(-1) + hc_sinkhorn_eps
    x = x / (x.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        x = x / (x.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    comb_mix = x

    layer_input = (residual.float() * pre_mix).sum(-2).bfloat16()

    return post_mix, comb_mix, layer_input


class TestMhcPreGemmSqrsumRace(unittest.TestCase):
    """Test that mhc_pre_gemm_sqrsum produces correct sqrsum values.

    The race condition bug manifests when m > 16 (multiple warps needed)
    and rows have different magnitudes.
    """

    def _check_sqrsum(self, m, hidden_size, hc_mult=4, tile_k=128):
        """Test sqrsum correctness for given dimensions."""
        from aiter.ops.mhc import mhc_pre_gemm_sqrsum
        from aiter import dtypes

        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 24
        hc_hidden_size = hc_mult * hidden_size

        torch.manual_seed(42)
        x_base = torch.randn(m, hc_hidden_size, dtype=dtypes.bf16, device="cuda")

        # Scale each row by a different factor to make sqrsum values distinct.
        # Row i is scaled by (i+1), so sqrsum[i] ~ (i+1)^2 * sqrsum_base.
        # If the race copies row (i%16)'s sqrsum to row i, the error is large.
        row_scales = torch.arange(1, m + 1, dtype=torch.float32, device="cuda")
        x = (x_base.float() * row_scales.unsqueeze(1)).bfloat16()

        fn = torch.randn(hc_mult3, hc_hidden_size, dtype=dtypes.fp32, device="cuda") * 0.01

        out_ref, sqrsum_ref = torch_ref_gemm_sqrsum(x, fn)

        split_k = 1
        out_pad = torch.empty(
            split_k, m, (hc_mult3 + 31) // 32 * 32,
            dtype=dtypes.fp32, device="cuda",
        )
        out_aiter = out_pad[:, :, :hc_mult3]
        sqrsum_aiter = torch.empty(split_k, m, dtype=dtypes.fp32, device="cuda")

        mhc_pre_gemm_sqrsum(out_aiter, sqrsum_aiter, x, fn, tile_k)
        torch.cuda.synchronize()

        sqrsum_aiter_flat = sqrsum_aiter.squeeze(0)

        abs_err = (sqrsum_aiter_flat - sqrsum_ref).abs()
        rel_err = abs_err / (sqrsum_ref.abs() + 1e-12)

        max_rel_err = rel_err.max().item()
        max_abs_err = abs_err.max().item()

        # With the race condition bug, rows 16+ may have sqrsum from row (i%16).
        # For row 16 (scale=17) vs row 0 (scale=1), rel error ~ |17^2 - 1^2|/17^2 ~ 99.7%
        # After fix, rel error should be < 1% (only numerical precision differences).
        bad_rows = (rel_err > 0.01).nonzero(as_tuple=True)[0]
        if len(bad_rows) > 0:
            print(f"\n  m={m}, hidden_size={hidden_size}")
            print(f"  Max rel error: {max_rel_err:.6f} (abs: {max_abs_err:.6f})")
            for row in bad_rows[:10].tolist():
                print(
                    f"    Row {row}: aiter={sqrsum_aiter_flat[row].item():.4f}, "
                    f"ref={sqrsum_ref[row].item():.4f}, "
                    f"rel_err={rel_err[row].item():.4f}"
                )
                if row >= 16:
                    alias_row = row % 16
                    print(
                        f"      -> row {row} % 16 = {alias_row}, "
                        f"ref[{alias_row}]={sqrsum_ref[alias_row].item():.4f} "
                        f"(race alias match: {abs(sqrsum_aiter_flat[row].item() - sqrsum_ref[alias_row].item()) < 0.01 * sqrsum_ref[alias_row].item().real})"
                    )

        self.assertLess(
            max_rel_err, 0.01,
            f"sqrsum max relative error {max_rel_err:.6f} exceeds 1% "
            f"for m={m}, hidden_size={hidden_size}. "
            f"Bad rows: {bad_rows.tolist()[:20]}"
        )

    def test_sqrsum_m64_h256(self):
        """m=64 spans 4 warps in one tile_m block (tile_m=64)."""
        self._check_sqrsum(m=64, hidden_size=256)

    def test_sqrsum_m64_h512(self):
        self._check_sqrsum(m=64, hidden_size=512)

    def test_sqrsum_m128_h256(self):
        """m=128 spans 2 tile_m blocks."""
        self._check_sqrsum(m=128, hidden_size=256)

    def test_sqrsum_m32_h1024(self):
        """m=32 spans 2 warps."""
        self._check_sqrsum(m=32, hidden_size=1024)

    def test_sqrsum_m64_h7168(self):
        """Real-world DeepSeek-V4 hidden size."""
        self._check_sqrsum(m=64, hidden_size=7168)

    def test_sqrsum_m128_h7168(self):
        self._check_sqrsum(m=128, hidden_size=7168)


class TestMhcPreFullPipeline(unittest.TestCase):
    """Test full mhc_pre pipeline (aiter vs torch reference)."""

    def _check_mhc_pre(self, m, hidden_size, hc_mult=4):
        from aiter.ops.mhc import mhc_pre
        from aiter import dtypes

        torch.manual_seed(42)

        residual = torch.randn(m, hc_mult, hidden_size, dtype=dtypes.bf16, device="cuda")
        row_scales = torch.arange(1, m + 1, dtype=torch.float32, device="cuda")
        residual = (residual.float() * row_scales.view(m, 1, 1)).bfloat16()

        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        hc_hidden_size = hc_mult * hidden_size
        fn = torch.randn(hc_mult3, hc_hidden_size, dtype=dtypes.fp32, device="cuda") * 0.01
        hc_scale = torch.randn(3, dtype=dtypes.fp32, device="cuda") * 0.1
        hc_base = torch.randn(hc_mult3, dtype=dtypes.fp32, device="cuda") * 0.1

        extra_args = dict(
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
        )

        post_ref, comb_ref, y_ref = torch_ref_mhc_pre(
            residual, fn, hc_scale, hc_base, **extra_args
        )

        post_aiter, comb_aiter, y_aiter = mhc_pre(
            residual, fn, hc_scale, hc_base, **extra_args
        )
        torch.cuda.synchronize()

        # post_mix comparison
        post_err = (post_aiter.squeeze(-1) - post_ref.squeeze(-1)).abs().max().item()
        self.assertLess(
            post_err, 0.05,
            f"post_mix max error {post_err:.6f} for m={m}, h={hidden_size}"
        )

        # comb_mix comparison
        comb_err = (comb_aiter - comb_ref).abs().max().item()
        self.assertLess(
            comb_err, 0.05,
            f"comb_mix max error {comb_err:.6f} for m={m}, h={hidden_size}"
        )

        # layer_input comparison (bf16 output, allow larger tolerance)
        y_err = (y_aiter.float() - y_ref.float()).abs().max().item()
        y_ref_scale = y_ref.float().abs().max().item() + 1e-12
        y_rel_err = y_err / y_ref_scale
        self.assertLess(
            y_rel_err, 0.05,
            f"layer_input max relative error {y_rel_err:.6f} for m={m}, h={hidden_size}"
        )

    def test_mhc_pre_m64_h512(self):
        self._check_mhc_pre(m=64, hidden_size=512)

    def test_mhc_pre_m128_h512(self):
        self._check_mhc_pre(m=128, hidden_size=512)

    def test_mhc_pre_m32_h1024(self):
        self._check_mhc_pre(m=32, hidden_size=1024)

    def test_mhc_pre_m64_h1024(self):
        self._check_mhc_pre(m=64, hidden_size=1024)


if __name__ == "__main__":
    unittest.main()
