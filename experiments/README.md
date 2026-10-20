# Experiments

This directory contains reproducible entry points for the thesis experiments.

## Recommended layout

- `run_racep_sine.py` — RACE-P smoke test on the synthetic Sine stream.
- `run_racep_*` — later dataset-specific reproductions.
- `run_criticisms_*` — MMD-Critic experiments.
- `compare_methods.py` — compare RACE-P, RACE-P + Criticism, ADWIN, and Page-Hinkley after results are saved.

Keep each script executable from the repository root:

```bash
python experiments/<script>.py
```

Do not put ad-hoc generated output files in this directory. Write logs to `logs/` and structured results to `results/`.
