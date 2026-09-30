# Q4MSE non-expert screen inspired by jayleaton

Reference inspected at b463237b014c1fd11915accc40ed6133691e5ea5:
https://github.com/jayleaton/glm53-tensorfold-spark (Apache-2.0 project).
The mechanism was ported from our existing dirty local TensorFold worktree:
qmm.py affine per-group MSE shrink search, weights.py opt-in projection
conversion, nonexpert.py and geometry/engine memory accounting. Original
worktree untouched; no patches from the external project copied in this port.
Default remains BF16. Explicit opt-in: TF_GLM_NONEXPERT=q4mse on both ranks.
Experts and latent kv_b remain untouched. This changes target numerics.

Actual full-model TP2, same source/config except projection mode, fc7:0.3,
context8192, same pinned weights/drafter. BF16 boot then Q4MSE boot, three
sequential requests each: Mia bench_decode.py prompts, 200 token limit,
temperature0, top_p1, thinking off, natural EOS, client first-content-to-EOF
metric. All completed200 tokens. First requests include cold/JIT effects.

| Median decode tok/s | BF16 | Q4MSE | Relative |
|---|---:|---:|---:|
| Hash-map prose | 23.48 | 28.73 | +22.3% |
| Counting | 53.61 | 67.33 | +25.6% |
| clamp_range code | 35.85 | 45.05 | +25.7% |

Drafted == serial complete text in all three Q4MSE tasks (200 tokens each).
This proves speculative parity in this format, NOT unchanged quality vs BF16.
Prose rounds changed60→65; counting25→25 and code36→36. Same round counts in
the latter two tasks support a faster round rather than more accepted tokens.
No broad quality qualification, logit/KL or downstream suite yet. No A/B/A.

Important: original production already measured34.61/62.80/51.01 tok/s on
these prompts (different deployed source/config). Q4MSE integration still loses
prose and code to production. Thus **do not promote this candidate** merely
because its own BF16 baseline improved. Audit production's loaded source,
projection mode and prepared weights before attributing the residual gap.

Compileall and git diff check passed. Experimental containers stopped;
unchanged original production containers restarted. Raw artifacts in
Proxmox/tensorfold-pool-results/recipe/mia-prompts-{bf16,q4mse}-fc7.json and
q4mse-serial-parity.json. GPU unit suite and broad numeric quality checks remain
outstanding; this is a measured experimental port, not a validated release.
