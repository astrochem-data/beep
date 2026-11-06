"""Rigid-body force-error decomposition of the geom_benchmark trajectory
benchmark (fragments + ranking_metric)."""
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from beep.core.trajectory_metrics import (
    decompose_force_errors_rigid, summarize_force_decomposition,
)
from beep.core import trajectory_workflow as tw
from beep.models.geom_benchmark import GeomBenchmarkConfig


# Two CO molecules, 1.13 A bonds, 3.5 A apart along x.
POS = np.array([
    [0.0, 0.0, 0.0], [1.13, 0.0, 0.0],
    [3.5, 0.0, 0.0], [4.63, 0.0, 0.0],
])
MASSES = np.array([12.011, 15.999, 12.011, 15.999])
FRAGS = [[0, 1], [2, 3]]


def _decomp(delta):
    return decompose_force_errors_rigid(
        np.asarray(delta)[None], POS[None], MASSES, FRAGS,
    )


def test_bond_stretch_error_is_purely_intramolecular():
    """Equal and opposite error along each CO bond (the functional wants a
    different bond length): zero net force, zero torque, all intramolecular."""
    delta = np.array([
        [-50.0, 0, 0], [50.0, 0, 0],
        [-50.0, 0, 0], [50.0, 0, 0],
    ])
    parts = _decomp(delta)
    assert np.allclose(parts["intermolecular"], 0.0, atol=1e-9)
    assert np.allclose(parts["torque"], 0.0, atol=1e-9)
    assert np.allclose(parts["intramolecular"], delta.ravel())


def test_rigid_translation_error_is_purely_intermolecular():
    """Same error vector on both atoms of one fragment: net force only."""
    delta = np.array([
        [0, 0, 0], [0, 0, 0],
        [10.0, 0, 0], [10.0, 0, 0],
    ])
    parts = _decomp(delta)
    assert np.allclose(parts["intermolecular"].reshape(2, 3)[0], 0.0)
    assert np.allclose(parts["intermolecular"].reshape(2, 3)[1], [20.0, 0, 0])
    assert np.allclose(parts["torque"], 0.0, atol=1e-9)
    assert np.allclose(parts["intramolecular"], 0.0, atol=1e-9)


def test_perpendicular_couple_is_torque_not_intramolecular():
    """Opposite forces perpendicular to a bond: a couple, i.e. torque,
    with zero net force and nothing intramolecular."""
    delta = np.array([
        [0, 5.0, 0], [0, -5.0, 0],
        [0, 0, 0], [0, 0, 0],
    ])
    parts = _decomp(delta)
    assert np.allclose(parts["intermolecular"], 0.0, atol=1e-9)
    assert np.allclose(parts["intramolecular"], 0.0, atol=1e-9)
    assert np.sqrt(np.mean(parts["torque"] ** 2)) > 0.0


def test_decomposition_rejects_partition_not_covering_molecule():
    with pytest.raises(ValueError, match="partition"):
        decompose_force_errors_rigid(
            np.zeros((1, 4, 3)), POS[None], MASSES, [[0, 1]],
        )


def test_summarize_force_decomposition_pools_systems():
    a = {"intermolecular": np.array([3.0]), "torque": np.array([0.0]),
         "intramolecular": np.array([1.0, 1.0])}
    b = {"intermolecular": np.array([4.0]), "torque": np.array([0.0]),
         "intramolecular": np.array([1.0, 1.0])}
    s = summarize_force_decomposition([a, b])
    assert np.isclose(s["rmsd_force_intermolecular"], np.sqrt((9 + 16) / 2))
    assert s["rmsd_torque"] == 0.0
    assert s["rmsd_force_intramolecular"] == 1.0
    assert np.isnan(summarize_force_decomposition([])["rmsd_torque"])


# --- config -------------------------------------------------------------

def _cfg(**kw):
    return GeomBenchmarkConfig(
        workflow="geom_benchmark", opt_dataset="ds",
        benchmark_structures=["co_2", "co_3"], **kw,
    )


def test_config_fragments_optional_and_ranking_default():
    cfg = _cfg()
    assert cfg.fragments is None
    assert cfg.ranking_metric == "intermolecular"


def test_config_fragments_must_cover_benchmark_structures():
    with pytest.raises(ValidationError, match="missing entries"):
        _cfg(fragments={"co_2": [[0, 1], [2, 3]]})


def test_config_fragments_must_be_disjoint():
    with pytest.raises(ValidationError, match="disjoint"):
        _cfg(fragments={"co_2": [[0, 1], [1, 2]], "co_3": [[0, 1]]})


def test_config_fragments_accepts_valid_partition():
    cfg = _cfg(fragments={"co_2": [[0, 1], [2, 3]], "co_3": [[0, 1], [2, 3], [4, 5]]},
               ranking_metric="total")
    assert cfg.fragments["co_3"][2] == [4, 5]


# --- ranking ----------------------------------------------------------

def _rmsd_df():
    return pd.DataFrame({"geom_gga_PBE": [0.10, 0.12], "geom_gga_BLYP": [0.20, 0.22]})


def test_ranking_uses_requested_force_key_and_relabels_weight():
    traj = {
        "pbe": {"rmsd_force": 100.0, "rmsd_force_intermolecular": 20.0},
        "blyp": {"rmsd_force": 50.0, "rmsd_force_intermolecular": 40.0},
    }
    weights = {"rmsd_eq": 0.0, "rmsd_force": 1.0}
    groups = {"geom_gga": ["PBE", "BLYP"]}
    by_total = tw.build_ranking_df(_rmsd_df(), traj, groups, weights, force_key="rmsd_force")
    by_inter = tw.build_ranking_df(
        _rmsd_df(), traj, groups, weights, force_key="rmsd_force_intermolecular",
    )
    assert by_total.index[0] == "blyp"
    assert by_inter.index[0] == "pbe"
    assert "rmsd_force_intermolecular" in by_inter.columns
    assert "z_rmsd_force_intermolecular" in by_inter.columns


def test_ranking_drops_functionals_without_decomposition():
    traj = {
        "pbe": {"rmsd_force": 100.0, "rmsd_force_intermolecular": 20.0},
        "blyp": {"rmsd_force": 50.0},
    }
    df = tw.build_ranking_df(
        _rmsd_df(), traj, {"geom_gga": ["PBE", "BLYP"]},
        {"rmsd_eq": 1.0, "rmsd_force": 1.0}, force_key="rmsd_force_intermolecular",
    )
    assert list(df.index) == ["pbe"]


# --- metrics integration (mocked record fetches) ------------------------

def _mol(geometry_A):
    mol = MagicMock()
    mol.symbols = ["C", "O", "C", "O"]
    mol.masses = MASSES
    mol.geometry = np.asarray(geometry_A) / tw.BOHR_TO_ANGSTROM
    return mol


def test_compute_metrics_adds_decomposition_when_fragments_given(test_logger):
    ref_g = np.zeros((4, 3))
    # DFT gradient error: pure bond stretch on both CO, in hartree/bohr
    err = np.array([[-1, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 0]]) * 1e-3
    steps = [{"molecule": _mol(POS), "energy_hartree": -1.0,
              "gradient_hartree_per_bohr": ref_g} for _ in range(2)]
    traj_data = {"co_2": steps}
    with patch.object(tw, "qcf") as qcf:
        qcf.fetch_sp_energy_gradient.return_value = (-1.0, ref_g + err)
        metrics, raw = tw.compute_per_method_trajectory_metrics(
            traj_data, {"co_2": MagicMock()}, ["pbe"], test_logger,
            fragments_per_struct={"co_2": FRAGS},
        )
    m = metrics["pbe"]
    assert m["rmsd_force"] > 0
    assert np.isclose(m["rmsd_force_intermolecular"], 0.0, atol=1e-9)
    assert np.isclose(m["rmsd_torque"], 0.0, atol=1e-9)
    assert np.isclose(m["rmsd_force_intramolecular"], m["rmsd_force"])
    assert "delta_intermolecular" in raw["pbe"]


def test_compute_metrics_without_fragments_is_unchanged(test_logger):
    ref_g = np.zeros((4, 3))
    steps = [{"molecule": _mol(POS), "energy_hartree": -1.0,
              "gradient_hartree_per_bohr": ref_g}]
    with patch.object(tw, "qcf") as qcf:
        qcf.fetch_sp_energy_gradient.return_value = (-1.0, ref_g + 1e-3)
        metrics, raw = tw.compute_per_method_trajectory_metrics(
            {"co_2": steps}, {"co_2": MagicMock()}, ["pbe"], test_logger,
        )
    assert "rmsd_force_intermolecular" not in metrics["pbe"]
    assert set(raw["pbe"]) == {"delta_e_per_atom_meV", "delta_force_meV_per_A"}


def test_compute_metrics_skips_bad_partition_with_warning(test_logger, caplog):
    ref_g = np.zeros((4, 3))
    steps = [{"molecule": _mol(POS), "energy_hartree": -1.0,
              "gradient_hartree_per_bohr": ref_g}]
    with patch.object(tw, "qcf") as qcf, caplog.at_level(logging.WARNING, logger="beep"):
        qcf.fetch_sp_energy_gradient.return_value = (-1.0, ref_g + 1e-3)
        metrics, _ = tw.compute_per_method_trajectory_metrics(
            {"co_2": steps}, {"co_2": MagicMock()}, ["pbe"], test_logger,
            fragments_per_struct={"co_2": [[0, 1]]},
        )
    assert "rmsd_force_intermolecular" not in metrics["pbe"]
    assert "fragment partition unusable" in caplog.text
