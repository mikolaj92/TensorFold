# Applying Mia's GB10 findings to our TensorFold GLM recipe

Reference: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
Inspected checkout: `674155dec2f2f62bb879801b5ce2cfc759a0bebf`.
This file records transfer candidates, not measured improvements.

## Concrete differences discovered

1. Mia defaults to DFlash2 with 7 proposed tokens. Our TensorFold EXL3 `auto`
   resolves to `fc5:0.3`: at most 5 proposals, stopped by cumulative confidence.
   These policies need separate measurement on identical prompts. Do not change
   the default merely because seven wins on a different drafter/engine.
2. Mia explicitly optimizes non-expert projections, with FP8 groups and BF16
   large-M KDA prefill. Their 2026-09-26 report gives about 4.7–4.9% shorter
   decode cycles for FP8=all vs its own prior production baseline. Their numeric
   report notes a small real change: text KL 25–30% above stock-pair controls.
   This is not bitwise-preserving quantization and needs quality validation.
3. The speed-boost preset uses a matched target dense overlay and 6-bpw EXL3
   drafter; our target and runtime-quantized drafter are different. Changing a
   target checkpoint is an accuracy/weight change, not merely a serving flag.
4. Mia separates prefill from decode and uses prose, structured and code tasks.
   Our earlier 128-token concurrency test measures whole-wave throughput. It
   demonstrates queueing but is not an equivalent performance comparison.
5. Their vLLM path batches requests; our current GLM TensorFold engine serializes
   them. Matching multi-request throughput needs a scheduler/engine change,
   not adoption of a DFlash depth flag.

## First transfer experiment (no copied source)

Use the existing recipe and identical target/drafter revisions. Compare startup
policies `auto`, `f7`, and `fc7:0.3` using a small isolated harness that
sets the existing engine default before loading the model. The current CLI
has `--mtp-drafts`, not a `--draft-policy` flag; do not invent one or assume
an integer MTP depth selects the same DFlash2 policy. Include draft=false as
control. Measure 512-token prose/structured/code replies, three repetitions,
TTFT and decode time separately, at short and 8k prompts. Check final token IDs
or response text against serial and count committed tokens per verify round.
Do not use /health's zero draft counters as an acceptance measurement.

Next: review our uncommitted `TF_GLM_NONEXPERT` projection work independently.
Do not silently transplant the dirty worktree. Start with exact BF16 baseline,
then compare projection formats and peak memory, logits/KL and downstream
known-answer behavior, with speculative parity **within each quantization**.
Only promote formats that provide measured full-model benefit.

## Licensing

Mia's current recipe/source is AGPL-3.0; older contributions before 2026-09-07
are described as MIT. TensorFold's own license must remain intact. Findings and
independent implementations can be discussed here with attribution; copying
current implementation requires a file/history-specific license review and
compatible distribution. No source from Mia has been copied into this branch.

## Status

Reference inspected, concrete policy and projection differences identified.
No new serving default, checkpoint or production deployment changed yet.
No claim that this branch reproduces Mia's reported throughput.
