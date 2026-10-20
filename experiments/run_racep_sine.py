"""RACE-P smoke test on the public Sine stream example.

Run from repository root:
    python experiments/run_racep_sine.py
"""

from river import forest
from protree.data.stream_generators import Sine
from protree.detectors import RaceP
from protree.explainers import APete


BLOCK_SIZE = 1000
DRIFT_POSITIONS = [28_000, 52_000, 70_000]
N_BLOCKS = 100


def main() -> None:
    model = forest.ARFClassifier()

    detector = RaceP(
        model=model,
        prototype_selector=APete,
        prototype_selector_kwargs={"alpha": 0.01},
        measure="prototype_reassignment_impact",
        strategy="total",
    )

    stream = Sine(drift_position=DRIFT_POSITIONS)

    for i in range(N_BLOCKS):
        x_block, y_block = zip(*stream.take(BLOCK_SIZE))

        for x, y in zip(x_block, y_block):
            model.learn_one(x, y)

        detector.update(x_block, y_block)

        if detector.drift_detected:
            print(f"{i * BLOCK_SIZE}) Drift detected!")


if __name__ == "__main__":
    main()
