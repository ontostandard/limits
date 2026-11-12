# Experiments of 12 November 2026

Preregistrations are kept exactly as written before each run (in Russian, our working language). Results: see the main README.

- `PREREG_probe_nd_2026-11-12.md`, `probe_nd.py`: linear probe for "not enough data" on two open 7B models.
- `PREREG_K1K2_2026-11-12.md`, `k2.py`: controls (text baselines, a 0.5B model, readout of the spoken answer).
- `PREREG_readout_nd_2026-11-12.md`, `readout_nd.py`: LoRA on the blocks above the probe layer, two seeds.
- `PREREG_ablate_nd_2026-11-12.md`: ablation of the probe subspace (the `--ablate` block in `readout_nd.py`).

The training set of 2,000 synthetic claims and the 201 real test claims with their key are not published, so the test keeps measuring.
