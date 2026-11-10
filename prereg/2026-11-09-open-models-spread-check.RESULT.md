# Result: open models spread check (pre-registration of 2026-11-09)

Run 2026-11-09 on a local RTX 4070 Laptop 8 GB, unsloth NF4 checkpoints, greedy, one pass. Raw answers in results/.

| model | false "impossible" (of 15) | impossible found, right limit (of 6) | possible kept (of 15) | need data noticed (of 4) |
| --- | --- | --- | --- | --- |
| OLMo-2 1B, ONTO method (RL), reference | 2 | 4 | 12 | 2 |
| Qwen2.5-Coder-7B-Instruct | 9 | 3 | 7 | 0 |
| Mistral-7B-Instruct-v0.3 | 1 | 2 | 5 | 2 |

Falsifier: NOT fired. Qwen is 7 above the reference on false "impossible"; Mistral is 2 below on impossible found and answers "need data" on 9 of the 15 possible claims. Decision: grow the set to 100 claims, freeze and pre-register before the next runs.

Notes, not changing the score:
- Qwen also answered "impossible" on DEMO-D1-0 (a possible claim) with the limit name "luminous_efficiency", outside the listed names; the scorer counts it as unclear, so its false "impossible" is 9 by the rule and 10 by reading.
- The runner was fixed twice before the first answer was generated (device placement; a pre-quantized checkpoint's own config). No change after.
- 25 claims, one pass: this decided whether to invest, not what any model can do.
