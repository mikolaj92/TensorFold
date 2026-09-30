# Community recipe: GLM-5.3-Flash EXL3 on two GB10 hosts

**Experimental integration, not an upstream release or a qualified deployment.**
Fork: https://github.com/mikolaj92/TensorFold/tree/recipes/glm53-gb10-tp2

## Included changes

Base upstream 0.5.0: `9cd52ab4daba68ddd09be89be8f23ad43175e821`.
PR heads below are preserved as merge parents, including their original authors.
See `glm53-gb10-tp2-manifest.json` for exact source SHAs.

- #128: 512-row prompt attention/selection scratch.
- #129: account for actual EXL3 scratch allocations.
- #132: idle follower waits on a TCPStore bell instead of a GPU collective.
- #134: DFlash2 sliding-window attention and ring cache with snapshots.
- #127: decisions API and cache-isolation fixes.
- #140: skip invisible pool-score tiles and radix padding.

Integration changes: decisions ring the follower bell before the score header;
follower receives the header after waiting and dispatches both request types.
Selection keeps #128's row blocks and supplies each block's device position to
#140's visible radix scan. These fixes are essential; an automatic merge alone
would not provide working decisions + idle bell semantics.

Not included: #131 changes the MTP default; #133 changes configurable memory
reserve. No reduction of the default host reserve. Uncommitted local projection/
weight-loader patches from the previous deployment are **not** silently included.
Consequently old deployment timings and admitted context sizes do not qualify
this integrated branch.

## Prerequisites and immutable installation

Two Linux ARM64 GB10 machines, each with 128 GB unified RAM, NVIDIA container
support, compatible driver/CUDA/PyTorch, and a working direct link. Install the
same commit on both hosts. Pin container image **by digest**, not a floating tag;
record driver, torch, CUDA and NCCL versions. The existing locally built
`glm53-tensorfold:dev` image is a test environment, not a publicly reproducible
published image.

```bash
git clone https://github.com/mikolaj92/TensorFold.git
cd TensorFold
git checkout recipes/glm53-gb10-tp2
# Resolve once, record, and use this SHA on BOTH hosts; don't track branch updates.
git rev-parse HEAD
# For subsequent reproduction: git checkout <recorded-commit-sha>
python -m pip install .
```

Download these revisions on **each** host (EXL3 dependencies as in upstream's
[GLM recipe](glm-5.3-flash.md)); use the resulting local snapshot directories:

- Target `neko-legends/GLM-5.3-Flash-Uncensored-EXL3`, revision
  `07135ec082f8f11f7a71e4244a4e5167a0f96277`.
- Drafter `incoai/GLM-5.3-Flash-DFlash2`, revision
  `7d74cdd881ed7e32c31175984a67823127b66cfe`.

The target is an uncensored third-party checkpoint, not the original Z.ai release.
Check access, licenses and suitability before use.

## Start a separate test deployment first

Never run this alongside an already resident full GLM on the same two machines.
Do not overwrite production's source bind mount. Use a separate immutable source
checkout/image, stop the old model only in a planned test window, and preserve its
image, source and launch command for rollback.

Choose MASTER as rank 0's direct-link address reachable from rank 1. The following
is an initial **8192-token qualification window**, not a claim of maximum context.
Use a port distinct from production's rendezvous and HTTP ports.

```bash
export TARGET=/path/to/target/snapshot
export DRAFTER=/path/to/drafter/snapshot
export MASTER=10.0.0.1
export TF_GLM_DRAFT_RING=1
export TORCH_CUDA_ARCH_LIST=12.1
# Rank 1, start first:
tensorfold serve "$TARGET" --drafter "$DRAFTER" --tp 2 --rank 1 \
  --master "$MASTER" --master-port 29651 --context 8192 \
  --name GLM-5.3-Flash-EXL3 --no-update-check
# Rank 0, same environment and model paths/revisions:
tensorfold serve "$TARGET" --drafter "$DRAFTER" --tp 2 --rank 0 \
  --master "$MASTER" --master-port 29651 --context 8192 \
  --name GLM-5.3-Flash-EXL3 --host 0.0.0.0 --port 18888 --no-update-check
```

HTTP listens on all interfaces for the lab; restrict network access externally.
For chat, explicitly set thinking off, or a template-supported reasoning effort
(e.g. high); GLM's medium may render as Max (#117). Decisions disable thinking.
Raise context only after checking admission, host memory and resume behavior.
Do not assume the former patched deployment's 200000 window still fits here.

## Validation and publication gates

Integrated host smoke: **34 passed, 28 skipped**, selected decisions, geometry,
kept-state, ring and idle-bell suites on Mac. Skipped GPU/torch-dependent cases
are **not** successes. No integrated CUDA or full-model TP2 test has run yet.

Previous PR-level results (not integrated-branch results) are in
`../decisions-validation.md` and `../glm-visible-pools.md`.

Before calling this recipe qualified, run:

1. CPU suites and CUDA kernel/latent/ring/engine suites; report admission refusals.
2. Full EXL3 TP2 greedy chat before → decisions → continuation, identical to control;
   multiple labels, invalid requests, errors, repeated and alternating conversations.
3. DFlash2 drafting vs draft=false, explicit supported policies, graph replay,
   ring wrap and snapshot restore; record actually drafted/accepted token counters.
4. Short, medium, ~200k contexts **if admitted**; actual full-model latency, TTFT,
   prefill/decode, RAM, and selection vs sparse-attention profile.
5. Idle both ranks, first request after idle, repeated decisions/chat wakeups,
   follower failure handling; record GPU and CPU usage, not just health.
6. Qwen MLX regression on M4 Max for shared server/decisions changes; no DSA speedup claim.

Rollback: stop both experimental ranks, restart the saved old rank 1 then rank 0
using their unchanged source/image/options; verify health and a chat. No automatic
upstream update. Publish new measured SHAs as new recipe revisions, not as an
unexplained moving production branch.
