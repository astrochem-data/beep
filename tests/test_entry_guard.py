"""Dataset entries must not be reused for a different geometry (periodic workflows)."""
from types import SimpleNamespace

import pytest
from qcelemental.models import Molecule

from beep.core.entry_guard import check_entry_geometry, guard_reused_entries


def _mol(dz=0.0, symbols=("O", "H", "H")):
    return Molecule(symbols=list(symbols), geometry=[0, 0, 0, 1.8, 0, 0, -0.5, 1.7, dz],
                    fix_com=True, fix_orientation=True)


class _FakeDataset:
    def __init__(self, name, entries, optimization):
        self.name = name
        self._entries = entries
        self._opt = optimization

    @property
    def entry_names(self):
        return list(self._entries)

    def iterate_entries(self, entry_names=None):
        for n in entry_names or self._entries:
            m = self._entries[n]
            yield SimpleNamespace(name=n, initial_molecule=m) if self._opt else SimpleNamespace(name=n, molecule=m)


def test_identical_geometry_passes():
    check_entry_geometry(_mol(), _mol(), "npASW_01_X00_Y00", "CO_npASW_01_surface")


def test_moved_geometry_raises_and_names_the_option():
    with pytest.raises(ValueError, match="dataset_suffix"):
        check_entry_geometry(_mol(), _mol(dz=0.5), "npASW_01_X00_Y00", "CO_npASW_01_surface")


def test_different_atoms_raise():
    with pytest.raises(ValueError, match="different atoms"):
        check_entry_geometry(_mol(), _mol(symbols=("O", "H", "F")), "x", "ds")


@pytest.mark.parametrize("optimization", [True, False])
def test_guard_checks_only_reused_names(optimization):
    ds = _FakeDataset("CO_npASW_01", {"a": _mol(), "b": _mol(dz=0.3)}, optimization)
    guard_reused_entries(ds, [("a", _mol()), ("new", _mol(dz=9.0))], optimization)
    with pytest.raises(ValueError):
        guard_reused_entries(ds, [("b", _mol())], optimization)


def test_periodic_image_counts_as_same_geometry():
    cell = [[31.06, 0, 0], [0, 31.06, 0], [0, 0, 800.0]]
    a = _mol()
    shifted = Molecule(symbols=["O", "H", "H"],
                       geometry=[0, 0, 0, 1.8, 0, 0, -0.5 + 31.06 / 0.529177210903, 1.7, 0.0],
                       fix_com=True, fix_orientation=True)
    with pytest.raises(ValueError):
        check_entry_geometry(a, shifted, "x", "ds")
    check_entry_geometry(a, shifted, "x", "ds", cell_ang=cell, pbc=[True, True, False])
    with pytest.raises(ValueError):   # not periodic along z
        z = Molecule(symbols=["O", "H", "H"], geometry=[0, 0, 0, 1.8, 0, 0, -0.5, 1.7, 800.0 / 0.529177210903],
                     fix_com=True, fix_orientation=True)
        check_entry_geometry(a, z, "x", "ds", cell_ang=cell, pbc=[True, True, False])
