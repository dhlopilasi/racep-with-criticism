# AGENTS.md — RACE-P / MMD-Critic Thesis

## 1. Project goal

This repository is used to reproduce and extend the RACE-P research code for a thesis on concept drift in data streams.

Primary research path:

1. Reproduce the original RACE-P behavior.
2. Keep the original RACE-P implementation as the primary baseline.
3. Add MMD-Critic as a separate extension.
4. Compare `RACE-P` vs `RACE-P + Criticism`.
5. Use ADWIN and Page-Hinkley as external drift-detector baselines when appropriate.

Do not silently change the original RACE-P algorithm while reproducing the paper.

## 2. Environment

Preferred environment:

- Python: 3.11
- Virtual environment: `.venv-racep`
- Activate with:
  `source .venv-racep/bin/activate`

Always verify the active interpreter with:

```bash
which python
python --version
```

## 3. Compatibility patch already applied

Current source compatibility patch:

```diff
- np.in1d(...)
+ np.isin(...)
```

Location: `protree/explainers/tree_distance.py`.

This patch is for compatibility with modern NumPy and should be kept isolated from research-method changes.

## 4. Reproduction rules

- Treat the public RACE-P source as the reference implementation.
- Do not modify detector logic, thresholds, prototype selection, or metric formulas during the reproduction stage unless explicitly requested.
- Record every intentional code change in `results/changes.md`.
- Record package versions used for experiments.
- Never overwrite previous experiment outputs; use a new result/log filename or run ID.
- Long experiments must run in the terminal/tmux, not through an agent session that waits for completion.
- Agents should inspect code, prepare commands/scripts, debug failures, and analyze saved results.

## 5. Current RACE-P smoke test

The public example currently used for the smoke test is the Sine stream:

- `block_size = 1000`
- `Sine(drift_position=[28000, 52000, 70000])`
- model: River `ARFClassifier`
- prototype selector: `APete`
- `alpha = 0.01`
- measure: `prototype_reassignment_impact`
- strategy: `total`

Run:

```bash
python experiments/run_racep_sine.py
```

Save output to `logs/` when running a long/repeatable experiment.

Important: the three Sine drift positions are ground-truth positions configured in the generator. Detector output must be evaluated against ground truth; do not automatically label every detector signal as a true drift.

## 6. Research terminology

Keep these concepts distinct:

- Prototype: an actual data instance selected to summarize/represent the data.
- Criticism: an actual data instance selected because it indicates poor representation / mismatch between the data distribution and the prototype representation.
- MMD, MCD, PRI: separate prototype-change measures. Do not average them into one score unless a new method explicitly defines such a combination.
- Drift signal: `drift_detected == True`.
- Ground-truth drift: drift position defined by the benchmark/generator or otherwise established by the experiment.
- False alarm / missed drift: evaluation outcomes, not automatically author-stated limitations.

## 7. Thesis extension: MMD-Critic

The preferred first extension is additive rather than invasive:

```text
RACE-P detector
    -> drift signal + prototype representation

MMD-Critic branch
    -> criticism instances / representation mismatch information

Combined output
    -> richer explanation of detected change
```

Do not claim that criticisms detect true drift unless an experiment demonstrates it.

For a first ablation, compare:

- `RACE-P`
- `RACE-P + Criticism`

External baselines:

- `ADWIN`
- `Page-Hinkley`

## 8. Evaluation

For detection, keep these concepts separate:

- precision
- recall / detection rate
- false alarms
- missed drifts
- detection delay

For explanation, inspect:

- prototype sets before/after a detected change
- criticism instances
- whether criticisms identify poorly represented or local regions

Do not use false alarms or missed drifts as "paper limitations" unless the paper explicitly states that.

## 9. Output discipline

Every experiment should save:

1. configuration / parameters
2. package versions
3. random seed(s), if used
4. detector signals
5. ground-truth drift positions, if available
6. summary metrics
7. stdout/stderr log

Preferred naming:

`<method>_<dataset>_<run_id>.*`

Example:

`racep_sine_run001.log`
`racep_sine_run001.json`

## 10. Agent behavior

Before editing code:

1. Inspect the relevant source.
2. Explain what component is being changed.
3. Keep reproduction and extension changes separate.
4. Make the smallest change that tests the hypothesis.

When a long-running experiment is needed:

- generate the command/script;
- let the user run it in tmux/nohup;
- return later to analyze saved logs/results.

Never spend the agent session waiting for a long experiment unless explicitly requested.
