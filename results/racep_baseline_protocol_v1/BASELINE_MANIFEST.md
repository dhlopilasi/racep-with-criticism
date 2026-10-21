# RACE-P Baseline — Protocol v1

## Configuration
- Profile: smoke
- Seeds: 2001–2010
- Number of seeds: 10
- Scenarios per seed: 5
- Chunks per scenario: 30
- Chunk size: 1000
- Reference chunks: 10
- Evaluation chunks: 20–29
- RACE-P measure: prototype_reassignment_impact
- RACE-P strategy: total
- RACE-P const: 3.0
- Prototype selector: APete
- Prototype selector alpha: 0.01

## Repository
- Local HEAD: c0a7082d282d3ca652bce97a347c73d548bc34d7
- origin/main: a75f4e878e2b097a3e09059a2d28033515a241c6
- Experiment script SHA256:
  6ca9e69bc26c173133dcbda084ce8ac67a1012979fcce6d0df9e89c6d9b87315  experiments/thesis_rebuild/run_controlled_e2_e4.py

## Results
- Seed runs: 10/10 successful
- Scenario-seed combinations: 50
- Evaluated chunks: 500
- Global concept detection runs: 2/10
- Local concept detection runs: 4/10
- Mixed concept detection runs: 0/10
- Stationary false-alarm runs: 5/10
- Virtual-drift false-alarm runs: 3/10
- Mean detector update time: 1040.20 ms
- Mean RACE-P model update time: 459.24 ms

## Interpretation and limitations
- Native RACE-P alarm flags are evaluated independently of the P/C combination rule.
- The runner still computes criticism signals in parallel.
- Component timings are recorded separately; total pipeline time is not a pure RACE-P-only runtime measurement.
- This is a controlled synthetic-scenario baseline, not a claim of exact reproduction of every experiment in the paper.
- Results must be preserved unchanged for subsequent comparisons.

## Status
Frozen candidate baseline: RACE-P protocol_v1.
