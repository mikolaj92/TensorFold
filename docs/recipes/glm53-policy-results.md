# First Mia-inspired policy transfer: full GLM TP2

Actual two GB10 hosts, recipe source 51650c8 plus isolated policy launcher,
same EXL3/DFlash2 revisions and context 8192 as the integration run. Policy
set identically on both ranks before model load; no weights or kernels changed.
Run order f7, fc5:0.3, fc7:0.3; one boot each, two requests per task, greedy,
thinking off, 256 forced output tokens, short prompts. This is a preliminary
screen, not A/B/A or a matched reproduction of Mia's benchmark.

Client decode estimate is (completion_tokens - 1)/(last text chunk time - first
text chunk time). Speculative chunks can contain multiple tokens, so this is an
approximation, not engine decode accounting. TTFT ~0.22–0.26 seconds.

| Policy | Prose tok/s | Code tok/s | Structured tok/s |
|---|---:|---:|---:|
| fc5:0.3 (baseline) | 21.05 | 34.21 | 28.52 |
| f7 | 20.19 | 36.37 | 30.68 |
| fc7:0.3 | 21.32 | 36.78 | 31.74 |

fc7 vs fc5: approximately +1.3% prose, +7.5% code, +11.3% structured in
this small sample. f7 worsens prose. **Do not change production/default yet.**
Six complete response texts per arm match across all three policies. Serial
reference parity at these 256-token tasks has not been measured in this run;
prior 128-token concurrency tasks had serial/draft text parity.

The gain does not close the gap with Mia's other checkpoint/drafter/projection
stack. Next controlled work: repeated boots, longer/8k prompts, serial checks,
then independently validated projection quantization. No projection changes
were applied in this experiment. Experimental servers stopped and original
production containers restarted; health checked separately after boot.
