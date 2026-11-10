# Pre-registration: do open models differ on the 25 demo claims?

Written 2026-11-09, before any of these runs.

## Question
Do off-the-shelf open instruct models score differently from our OLMo-2 1B on the 25 demo claims? If they all score the same, a bigger set is not worth building.

## Runs
- Qwen2.5-Coder-7B-Instruct, 4-bit NF4, local RTX 4070 8 GB.
- One more open instruct model of 7-8B from a different family, same settings.
- Greedy decoding, max 256 new tokens, one pass. Prompt and parser: `examples/run_hf.py`, unchanged after the first answer is generated.
- The prompt lists the limit names the scorer accepts, so a model is not penalised for naming a law differently.
- Scored by `score.py`, verdict mode.

## Reference (already published in results/)
OLMo-2 1B, ONTO method (RL): false "impossible" 2 of 15, impossible found 4 of 6, need data 2 of 4.

## Falsifier
The set is NOT grown if, for both models, all three hold:
- false "impossible" within 1 of 2 (that is 1 to 3 of 15),
- impossible found within 1 of 4 (3 to 5 of 6),
- need data noticed within 1 of 2 (1 to 3 of 4).

Otherwise the models differ, and the set is grown to 100 claims, frozen and pre-registered before the next runs.

## Known limits
25 claims, one pass, one seed. This check decides whether to invest, not what any model can do.
