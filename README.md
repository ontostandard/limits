# ONTO Limits

**How often does a language model call something impossible when it isn't?** A small, real-world test of one failure that matters for fact-checking and AI safety: a false "impossible" on a real product or news claim, or a missed one when a physical limit really is broken.

Every claim here comes from a real product page, press release or news story, with a link to its source. A model reads the claim and answers one of three things:

- **impossible**, and which physical limit is broken (Betz, Carnot, first law of thermodynamics, luminous efficacy, Shockley–Queisser, speed of light)
- **possible**
- **need data**, when the text does not give enough to decide

The scorer needs no AI judge. "Impossible" counts only with the right limit. In grounded mode it counts only if the model's own call to a limits library actually returned a violation.

## Results on the 25 demo claims (open, reproduce them yourself)

| Model | False "impossible" (of 15 possible) | Impossible found, right limit (of 6) | Possible kept (of 15) | Need data noticed (of 4) |
| --- | --- | --- | --- | --- |
| OLMo-2 1B, ONTO method (RL) | **2** | **4** | **12** | 2 |
| OLMo-2 1B, ONTO method (SFT) | 3 | 4 | 12 | 1 |
| Qwen2.5-Coder-7B-Instruct | 9 | 3 | 7 | 0 |
| Mistral-7B-Instruct-v0.3 | 1 | 2 | 5 | 2 |

The two 7B models fail in opposite directions: Qwen calls 9 of 15 real, possible products impossible; Mistral almost never says impossible and answers "need data" on 9 of the 15 possible claims. 7B rows: 4-bit NF4, greedy, one pass, prompt in [`examples/run_hf.py`](examples/run_hf.py). 25 claims is a demo: it shows the method, not a ranking. The criteria were written before these runs: [`prereg/`](prereg).

Raw answers for every row are in [`results/`](results). Reproduce any row:

```bash
python score.py results/qwen2.5-coder-7b-instruct-nf4.json
```

## Results on the closed sets (aggregate numbers only)

Two larger sets we keep closed, so that they keep measuring. v1: 100 real claims (47 impossible, 36 possible, 17 lack the data to decide). v2: 101 real claims (29 / 44 / 28), never used for development.

**Frontier model, consumer chat, one pass, scored against the key with no AI judge:**

| Set · condition | Impossible found, right limit | False "impossible" on possible | "Need data" said correctly |
| --- | --- | --- | --- |
| v1 · no library | 44 of 47 | 0 of 36 | 11 of 17 |
| v2 · no library | 25 of 29 | 0 of 44 | 14 of 28 |
| v2 · with the limits library | 27 of 29 | 0 of 44 | 20 of 28 |

On v2 the library did not reliably help a frontier model: +2 impossible claims found, exact McNemar p = 0.625, our pre-registered criterion for "it helps" failed. What the frontier model lacks is not physics but saying "not enough data" when that is the true answer. On "need data" claims it also called 3 of 28 impossible without the library, 1 of 28 with it. One pass in a chat, not an API run with fixed versions; a repeated API run across several models is in progress.

**Smaller models, v1, judged by two blind AI judges with a third on disagreements:**

| Who answers | Impossible found, right limit (of 47) | False "impossible" (of 36) |
| --- | --- | --- |
| Qwen2.5-Coder 7B, through the library | 22 | 4 |
| Mistral 7B, forced to give a verdict | 16 | 3 |
| OLMo-2 1B, ONTO method | 15 | 6 |
| Mistral 7B, plain answer | 12 | 4 |
| Qwen2.5-Coder 7B, plain answer | 9 | 1 |

7B rows are 4-bit NF4. Mistral 7B called the library once in 100 claims, so its gain came from being made to give a verdict, not from the library.

**Correction, 12 November 2026.** An earlier version of this page said a frontier model found 42 of 47 without the library and 47 of 47 with it, and that "the library of limits is enough". Those frontier figures came from an agent setup that overstated the result. The clean runs above replace them.

## Test your own model

1. Give your model each `text` from [`data/claims.json`](data/claims.json). A prompt that works is in [`examples/run_hf.py`](examples/run_hf.py), which also runs any Hugging Face model end to end:

```bash
python examples/run_hf.py Qwen/Qwen2.5-Coder-7B-Instruct my_answers.json --4bit
```

2. Or save your own answers as JSON:

```json
{"answers": [
  {"id": "DEMO-D1-1", "answer": "impossible", "limit": "white_light"},
  {"id": "DEMO-D1-0", "answer": "possible", "limit": null}
]}
```

3. Run `python score.py my_answers.json`. Python 3, no dependencies for the scorer.

Limit names the scorer understands are listed in [`score.py`](score.py) (`GROUP_OF`). Names of the same law count as the same answer.

## Why this matters

Most work on LLM hallucination looks at models inventing facts. The mirror failure gets less attention: calling a real product a scam, or a working design a violation of physics. For fact-checking and for AI safety, a verdict should follow from a check the model actually ran, not from how the text sounds.

## What is here and what is not

| Here | Not here |
| --- | --- |
| 25 claims with sources, answer key, scorer, runner, raw results, reports, pre-registrations | training data, the limits library code, model weights, the closed 100-claim set |

Claims are short quotes from public pages, kept for research and linked to their source.

## Repository

```
data/claims.json        25 claims: id, text, source link
data/answer_key.json    answer and limit for each claim
score.py                judge-free scorer
examples/run_hf.py      run any Hugging Face model on the claims
results/                raw answers per model
reports/                dated lab reports
prereg/                 criteria written before each run, and their results
```

## About

[ONTO](https://ontostandard.org) is an independent AI research lab. Khakim Angenoev, founder. Contact: council@ontostandard.org

Code: MIT. Answer key and reports: CC BY 4.0.
