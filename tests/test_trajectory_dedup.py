"""RMSD-based trajectory frame dedup (compute-saving pre-filter) for the
geom_benchmark trajectory force workflow."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from beep.core.trajectory_workflow import _kabsch_rmsd, _dedup_trajectory_steps
from beep.core.trajectory_workflow import BOHR_TO_ANGSTROM

log = logging.getLogger("beep")


def _step(coords_ang):
    """Mock trajectory step: only .molecule.geometry (flat, Bohr) is used."""
    g = (np.asarray(coords_ang, float) / BOHR_TO_ANGSTROM).ravel()
    return {"molecule": SimpleNamespace(geometry=g),
            "energy_hartree": 0.0,
            "gradient_hartree_per_bohr": np.zeros((len(coords_ang), 3))}


BASE = [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]  # a CO-like 2-atom frame (Angstrom)


def test_kabsch_zero_for_identical():
    P = np.random.RandomState(0).randn(5, 3)
    assert _kabsch_rmsd(P, P) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_invariant_to_rigid_motion():
    P = np.random.RandomState(1).randn(5, 3)
    th = 0.9
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Q = P @ R.T + np.array([4.0, -2.0, 1.5])     # rotate + translate
    assert _kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_positive_for_different():
    P = np.random.RandomState(2).randn(5, 3)
    Q = np.random.RandomState(3).randn(5, 3)
    assert _kabsch_rmsd(P, Q) > 0.1


def test_dedup_disabled_by_default_returns_all():
    steps = [_step(BASE), _step(BASE), _step(BASE)]
    for thr in (None, 0, 0.0):
        assert _dedup_trajectory_steps(steps, thr, 1, log, "x") is steps


def test_dedup_collapses_near_identical_keeps_distinct():
    # f0 and f1 identical; f1b a hair off (<0.02); f2 clearly displaced
    f0 = _step(BASE)
    f1 = _step(BASE)
    f1b = _step([[0.0, 0.0, 0.0], [1.135, 0.0, 0.0]])    # ~0.005 A change
    f2 = _step([[0.0, 0.0, 0.0], [1.60, 0.0, 0.0]])      # ~0.47 A change
    kept = _dedup_trajectory_steps([f0, f1, f1b, f2], 0.02, 1, log, "x")
    assert len(kept) == 2            # one representative of the cluster + f2
    assert kept[0] is f0 and kept[-1] is f2


def test_dedup_preserves_order():
    steps = [_step([[0, 0, 0], [1.13 + 0.3 * i, 0, 0]]) for i in range(5)]
    kept = _dedup_trajectory_steps(steps, 0.02, 1, log, "x")
    assert kept == steps             # every frame distinct -> all kept, in order


def test_dedup_min_frames_floor():
    # all frames within threshold -> would collapse to 1; floor of 3 keeps 3
    steps = [_step([[0, 0, 0], [1.13 + 0.001 * i, 0, 0]]) for i in range(6)]
    kept1 = _dedup_trajectory_steps(steps, 0.02, 1, log, "x")
    assert len(kept1) == 1
    kept3 = _dedup_trajectory_steps(steps, 0.02, 3, log, "x")
    assert len(kept3) == 3
    assert kept3[0] is steps[0]      # first frame always kept


def test_config_accepts_dedup_fields():
    from beep.models.geom_benchmark import GeomBenchmarkConfig
    c = GeomBenchmarkConfig(
        workflow="geom_benchmark", opt_dataset="d",
        benchmark_structures=["A"],
        trajectory_dedup_rmsd=0.02, trajectory_dedup_min_frames=2,
    )
    assert c.trajectory_dedup_rmsd == 0.02
    assert c.trajectory_dedup_min_frames == 2
    # default off
    d = GeomBenchmarkConfig(workflow="geom_benchmark", opt_dataset="d",
                            benchmark_structures=["A"])
    assert d.trajectory_dedup_rmsd is None
    assert d.trajectory_dedup_min_frames == 1
