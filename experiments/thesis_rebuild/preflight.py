"""Check that the active environment exposes the APIs required by the bundle.

Run from the Protree repository root after activating .venv-racep:
  python -m experiments.thesis_rebuild.preflight
"""
from __future__ import annotations

import inspect
from importlib import metadata


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-found-by-package-metadata"


def main() -> None:
    import river
    from river import drift, forest
    import protree
    from protree.criticism.mmd_critic import MMDCritic
    from protree.detectors import RaceP
    from protree.explainers import APete

    print(f"Python package river: {_version('river')} (module={getattr(river, '__version__', 'unknown')})")
    print(f"Python package protree: {_version('protree')} (module={getattr(protree, '__version__', 'unknown')})")
    print(f"ARFClassifier: {inspect.signature(forest.ARFClassifier)}")
    print(f"PageHinkley: {inspect.signature(drift.PageHinkley)}")
    print(f"RaceP: {inspect.signature(RaceP)}")
    print(f"APete: {inspect.signature(APete)}")
    print(f"APete.select_prototypes: {inspect.signature(APete.select_prototypes)}")
    print(f"MMDCritic: {inspect.signature(MMDCritic)}")
    print(f"MMDCritic.select: {inspect.signature(MMDCritic.select)}")

    required = inspect.signature(MMDCritic.select).parameters
    if "prototype_indices" not in required:
        raise SystemExit("ERROR: local MMDCritic.select lacks the prototype_indices argument required by E1.")
    racep_parameters = inspect.signature(RaceP).parameters
    needed_racep = {"model", "prototype_selector", "measure", "strategy", "grace_period", "const"}
    missing = needed_racep - set(racep_parameters)
    if missing:
        raise SystemExit(f"ERROR: local RaceP API is missing parameters required by the runner: {sorted(missing)}")
    print("Preflight passed. This confirms imports/signatures only; run the smoke stream before larger experiments.")


if __name__ == "__main__":
    main()
