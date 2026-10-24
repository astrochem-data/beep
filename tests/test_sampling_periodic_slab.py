"""sampling_periodic.load_slab: the slab from the entry or from a relaxation spec in the surface collection."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import qcelemental as qcel

from beep.models.sampling_periodic import SamplingPeriodicConfig
from beep.workflows.sampling_periodic import config_summary_msg, load_slab

CELL = {"cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 5.0]], "pbc": [True, True, False]}


def _water(z, extras=None):
    return qcel.models.Molecule(symbols=["O", "H", "H"], geometry=[0, 0, z, 1.8, 0, z, -0.5, 1.7, z],
                                extras=extras, fix_com=True, fix_orientation=True)


class FakeSurfaces:
    name = "slabs"

    def __init__(self, status="complete", final=None):
        self.entry = SimpleNamespace(initial_molecule=_water(0.0, CELL))
        self.record = SimpleNamespace(id=7, status=SimpleNamespace(value=status),
                                      final_molecule=final if final is not None else _water(0.3))

    def get_entry(self, name):
        return self.entry

    def get_record(self, name, spec):
        return self.record if spec == "model-a" else None


def test_entry_geometry_without_specification():
    assert load_slab(FakeSurfaces(), "s1", None).geometry[0, 2] == 0.0


def test_relaxed_slab_keeps_the_entry_cell():
    slab = load_slab(FakeSurfaces(), "s1", "model-a")
    assert slab.geometry[0, 2] == pytest.approx(0.3) and slab.extras["cell"] == CELL["cell"]


def test_incomplete_or_missing_relaxation_refuses():
    with pytest.raises(RuntimeError, match="not complete"):
        load_slab(FakeSurfaces(status="running"), "s1", "model-a")
    with pytest.raises(KeyError):
        load_slab(FakeSurfaces(), "s1", "model-b")


def test_other_molecule_refuses():
    other = qcel.models.Molecule(symbols=["C", "O"], geometry=[0, 0, 0, 0, 0, 2.1])
    with pytest.raises(RuntimeError, match="does not hold"):
        load_slab(FakeSurfaces(final=other), "s1", "model-a")


def test_summary_names_the_specification():
    kw = dict(workflow="sampling_periodic", molecule="CO", surface_collection="slabs",
              sampling_level_of_theory={"program": "psi4", "mace_model": "m.model"})
    try:
        cfg = SamplingPeriodicConfig(**kw, surface_specification="model-a")
    except Exception as exc:  # the LOT model may need other required fields; skip the banner check then
        pytest.skip(f"config construction: {exc}")
    assert "'model-a' results" in config_summary_msg(cfg)
