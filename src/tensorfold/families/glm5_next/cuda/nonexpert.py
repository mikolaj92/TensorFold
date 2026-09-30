"""Which EXL3 BF16 linears ``TF_GLM_NONEXPERT`` stores as 4-bit, and what that does to the startup byte count."""

from __future__ import annotations

import os

# ``q4`` / ``stack`` / the EXL3 ``lm_head`` in ``weights.load``. Routed experts, norms, convs and ``kv_b_proj`` stay.
_Q4_WEIGHT = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "f_a_proj.weight", "g_a_proj.weight", "b_proj.weight",
    "f_b_proj.weight", "g_b_proj.weight", "o_proj.weight", "q_a_proj.weight", "kv_a_proj_with_mqa.weight",
    "q_b_proj.weight", "indexer.wk.weight", "indexer.weights_proj.weight", "indexer.wq_b.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight", "eh_proj.weight", "lm_head.weight",
)


def nonexpert_mode() -> str:
    """How EXL3 stores the linears that are BF16 in the checkpoint: ``bf16`` (upstream), ``q4``, or ``q4mse``."""

    mode = os.environ.get("TF_GLM_NONEXPERT", "bf16").strip().lower()
    if mode not in ("bf16", "q4", "q4mse"):
        raise ValueError("TF_GLM_NONEXPERT must be bf16, q4, or q4mse")
    return mode


def quantizes_nonexpert(name: str) -> bool:
    """True when ``load`` replaces this EXL3 BF16 matrix with ``quantize4``."""

    if ".mlp.experts." in name or "embed_tokens" in name or "kv_b_proj" in name:
        return False
    return name == "lm_head.weight" or name.endswith(_Q4_WEIGHT)


def nonexpert_bytes(name: str, info: dict, shape: list[int], total: int, host: int) -> tuple[int, int]:
    """Startup bytes after ``TF_GLM_NONEXPERT`` quantizes a matrix: one padded 4-bit copy, no second draft head."""

    if nonexpert_mode() == "bf16" or not quantizes_nonexpert(name) or info.get("dtype") not in ("BF16", "F16"):
        return total, host
    if len(shape) != 2 or shape[-1] % 64:
        return total, host
    n = ((shape[0] + 127) // 128) * 128
    return n * shape[1] * 9 // 16, host
