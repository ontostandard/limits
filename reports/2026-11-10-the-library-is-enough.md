# The library is enough. The reader isn't.

*What training a 1B open model to say "impossible" taught us about tool use, hallucination and reading.*

> **Correction, 12 November 2026.** The frontier-model figures in this report came from an agent setup that overstated the result. In a clean, pre-registered rerun on the 101-claim closed set, the library added 2 found claims to a frontier model (27 vs 25 of 29, McNemar p = 0.625), so "the library is enough" is not supported for frontier readers. The finding about the small model's floor of false "impossible" stands. Current numbers: [README](../README.md).


Published on Medium on 10 November 2026: https://medium.com/@ontostandard/the-library-is-enough-the-reader-isnt-939fa7bb0f4f

A product page sells a 500 W wind turbine with a 0.4 m rotor. A press release says a pilot plant takes 1.6 kW in and gives 11.8 kW out. A flashlight promises 100,000 lumens from a 50 W chip. You do not need an expert to check any of these. You need a few lines of arithmetic and one physical limit for each.

We spent the last months on a narrow question. Can a small open language model learn to do that check by itself, and say "impossible" only when the check says so?

## Why this matters

Most talk about LLM hallucination is about models inventing facts. The mirror failure gets less attention: a model that calls something impossible when it is not. In fact-checking, that is a false accusation. A model that labels a real product as a scam, or a working design as a violation of physics, does as much damage as one that believes everything. For AI safety the question is the same in both directions: does the model's verdict follow from evidence it actually checked, or from how the text sounds?

## What a check looks like

Take the turbine. A 0.4 m rotor sweeps about 0.126 m². At a strong wind of 12 m/s (our assumption, the page does not say), the wind carries about 133 W through that area. No turbine can take more than 59.3% of it. That is the Betz limit. So the ceiling is about 79 W, and 500 W is over six times too much. The flashlight is simpler: no light source can beat 683 lumens per watt, so 50 W gives at most about 34,000 lumens. The pilot plant breaks the first law of thermodynamics unless it has a fuel the release does not mention, and then the claim is "missing data", not "impossible".

That last case is the hard part. The honest answer is often not yes or no. It is "the text does not give me enough to decide".

## The setup

The model is OLMo-2 1B, a fully open model: weights, data and training code are public. It reads a real claim and writes up to six lines of code that pull the numbers out of the text and attach units to them. Then it calls a library of physical limits: Betz, Carnot, the first law, luminous efficacy and others. The library answers whether the claim is within limits and by what factor it exceeds one.

The model gives one of three verdicts: impossible (which limit, how many times over), possible, or missing data. One rule is built into the scoring, not the prompt: "impossible" counts only if a library call actually returned a violation. The model cannot argue its way into it. This is what we mean by grounded reasoning: the verdict has to point at a check that ran.

## How we keep ourselves honest

Before every training run we write down two things: the threshold that means "accepted", and the result that would prove the direction wrong. We develop on a set of 100 real claims. We accept or reject on a separate set of 101 claims that we never open to look at errors. Every recipe runs with two random seeds. Negative results go into the record as results. This is slower than tuning on the test set, and it is the only way the numbers below mean anything.

## The result that matters

On the closed set we tried six recipes: supervised fine-tuning, reinforcement learning with a sandbox reward, training on repaired traces, and guidance in the prompt. Across all of them, the 1B model wrongly says "impossible" on 7 to 9 of 44 claims that are actually possible. The latest recipe averages 8.5 of 44. The floor does not move.

Then we gave the same real texts and the same library to a strong reader, a frontier model doing only the reading. It made 1 false "impossible" on 79 real possible claims. That is 1.3%.

Same library. Same rule. The difference is the reader.

## Where the small model fails

We went through every false "impossible" on the development set by hand. Three were unit confusions: kilowatt-hours per month read as power, an area read as an amount of light. One was a misread percentage. Three were plain misreadings of the sentence. In every case the correct limit was available. The model chose the right tool and fed it the wrong numbers.

A dimension check after the fact catches some of these. It also throws away correct answers the model reached by luck, so it is not a fix. Then we audited our own training data and found that 3 to 11% of the worked examples had a dimension error in them. We had been teaching the shape of a solution, not its meaning. The model learned the shape very well.

## What this says about tool use

The usual story about small language models is that they lack knowledge, and tools fill the gap. Our data says something narrower. A library of limits is enough to decide these claims. It is enough only for a reader who knows what a kilowatt-hour is, which number in the sentence is the input and which is the output. Tool use does not replace reading. It moves the failure from "what is the answer" to "what does the text say".

## What we test next

Our working idea is knowledge density. A small model pretrained on denser, more technical text should read claims better at the same size. We are testing it now: SmolLM2-360M against Gemma 3 270M on the same task, with the success criterion written before the run. We will publish the result whichever way it goes.

## What we are opening

On 12 November we publish ONTO Limits on GitHub: a demo set of 25 real claims with links to their sources, never used in training, and a scorer that needs no AI judge. The scorer reads the library calls the model actually made, so anyone can check a model's verdicts without trusting another model. The training corpus, the library code and the weights stay closed for now.

## Try it on your model

The repository opens on 12 November. If you build or evaluate language models, follow ONTO on [X](https://x.com/ontostandards) or [LinkedIn](https://www.linkedin.com/company/ontostandard/) and you will get the link the day it goes live. Run the 25 claims through your model and see how often it says "impossible" when it shouldn't.

*ONTO is an independent AI research lab mapping what language models can't do. ontostandard.org*
