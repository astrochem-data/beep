"""
Trajectory benchmark metrics — pure computation, no QCFractal deps.

Energies are reported in meV/atom; gradient components (forces) in meV/Å.
This matches the units used in MLP-validation literature (Bovolenta
et al. 2025, A&A, Fig C.1) and makes the metrics directly comparable
across cluster sizes.

Conversion factors are taken from CODATA 2018 / qcelemental.
"""
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


HARTREE_TO_MEV = 27211.386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_MEV_PER_A = HARTREE_TO_MEV / BOHR_TO_ANGSTROM


def per_step_deltas(
    ref_energies,
    ref_gradients,
    dft_energies,
    dft_gradients,
    n_atoms: int,
) -> Dict[str, np.ndarray]:
    """Compute per-trajectory-step (DFT − reference) deltas in MLP units.

    Parameters
    ----------
    ref_energies, dft_energies : array-like, shape (n_steps,)
        Total energies in hartree at each trajectory step.
    ref_gradients, dft_gradients : array-like, shape (n_steps, n_atoms, 3)
        Cartesian gradients in hartree/bohr at each trajectory step.
    n_atoms : int
        Number of atoms in the supermolecule, for per-atom energy
        normalization.

    Returns
    -------
    dict with keys:
      ``delta_e_per_atom_meV``  — shape (n_steps,), meV/atom
      ``delta_force_meV_per_A`` — shape (n_steps, n_atoms, 3), meV/Å
    """
    ref_e = np.asarray(ref_energies, dtype=float)
    dft_e = np.asarray(dft_energies, dtype=float)
    ref_g = np.asarray(ref_gradients, dtype=float)
    dft_g = np.asarray(dft_gradients, dtype=float)

    if ref_e.shape != dft_e.shape:
        raise ValueError(
            f"ref_energies shape {ref_e.shape} != dft_energies shape {dft_e.shape}"
        )
    if ref_g.shape != dft_g.shape:
        raise ValueError(
            f"ref_gradients shape {ref_g.shape} != dft_gradients shape {dft_g.shape}"
        )
    if n_atoms <= 0:
        raise ValueError(f"n_atoms must be > 0, got {n_atoms}")

    delta_e_per_atom_meV = (dft_e - ref_e) * HARTREE_TO_MEV / n_atoms
    delta_force_meV_per_A = (dft_g - ref_g) * HARTREE_PER_BOHR_TO_MEV_PER_A
    return {
        "delta_e_per_atom_meV": delta_e_per_atom_meV,
        "delta_force_meV_per_A": delta_force_meV_per_A,
    }


def summarize_method_metrics(
    per_system_deltas: Iterable[Dict[str, np.ndarray]]
) -> Dict[str, float]:
    """Aggregate per-system delta arrays for one functional and summarize.

    The energy and force deltas are concatenated across systems
    (a trajectory of length N_i contributes N_i energy points and
    3 · N_i · n_atoms_i force-component points), then reduced to
    root-mean-square deviation (RMSD).

    RMSD vs the reference is the right metric here: same mathematical
    form as the equilibrium-geometry ``compare_rmsd``; outlier-sensitive
    (a single bad gradient point — the failure mode that matters for
    geometry optimization — surfaces clearly).

    Force RMSD is taken over every Cartesian component (matching the
    Bovolenta 2025 Fig C.1 convention). Energy RMSD is computed for
    completeness — the geom_benchmark workflow does not use it (absolute
    energies aren't meaningful for geometry ranking; use the
    energy_benchmark workflow for relative-energy comparison).
    """
    per_system_deltas = list(per_system_deltas)
    if not per_system_deltas:
        return {
            "rmsd_energy": float("nan"),
            "rmsd_force": float("nan"),
            "n_energy_points": 0,
            "n_force_components": 0,
        }

    all_de = np.concatenate(
        [np.asarray(d["delta_e_per_atom_meV"]).ravel() for d in per_system_deltas]
    )
    all_df = np.concatenate(
        [np.asarray(d["delta_force_meV_per_A"]).ravel() for d in per_system_deltas]
    )

    return {
        "rmsd_energy": float(np.sqrt(np.mean(all_de ** 2))),
        "rmsd_force": float(np.sqrt(np.mean(all_df ** 2))),
        "n_energy_points": int(all_de.size),
        "n_force_components": int(all_df.size),
    }


def decompose_force_errors_rigid(
    delta_force,
    positions_A,
    masses,
    fragment_atom_indices,
) -> Dict[str, np.ndarray]:
    """Split per-step force errors into per-fragment rigid-body parts and
    the intramolecular remainder.

    Parameters
    ----------
    delta_force : array-like, shape (n_steps, n_atoms, 3)
        DFT − reference force error at every step (any unit; meV/Å here).
    positions_A : array-like, shape (n_steps, n_atoms, 3)
        Atomic positions at every step, Å.
    masses : array-like, shape (n_atoms,)
    fragment_atom_indices : sequence of sequences of int
        Disjoint atom-index lists, one per molecule.

    The forces that atoms of one fragment exert on each other cancel in
    the fragment's net force and torque (Newton's third law), so per
    fragment and step

    * ``intermolecular`` = Σ_i δ_i, the net force error on the fragment
      (3 components), purely intermolecular whatever the bond-stretch
      error;
    * ``torque``         = Σ_i (r_i − r_COM) × δ_i (3 components,
      force unit × Å), also purely intermolecular;
    * ``intramolecular`` = δ with the fragment's rigid translation and
      rotation projected out (``rigid_body_split``), i.e. the force
      pattern with zero net force and zero torque that only changes
      internal coordinates (3 components per atom).

    Returns a dict of flat arrays pooled over steps and fragments.
    Definitions match ``decompose_force_errors`` in
    ``normal_mode_sampling`` so nm_sampling and trajectory numbers are
    comparable.
    """
    from .normal_mode_sampling import rigid_body_split

    delta = np.asarray(delta_force, dtype=float)
    pos = np.asarray(positions_A, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if delta.ndim != 3 or delta.shape[-1] != 3:
        raise ValueError(f"delta_force must be (n_steps, n_atoms, 3), got {delta.shape}")
    if pos.shape != delta.shape:
        raise ValueError(f"positions shape {pos.shape} != delta shape {delta.shape}")
    n_atoms = delta.shape[1]
    covered = sorted(a for frag in fragment_atom_indices for a in frag)
    if covered != list(range(n_atoms)):
        raise ValueError(
            f"fragment partition covers atoms {covered[:5]}... ({len(covered)}), "
            f"molecule has {n_atoms}"
        )

    inter, torque, intra = [], [], []
    for step_delta, step_pos in zip(delta, pos):
        for frag in fragment_atom_indices:
            frag = list(frag)
            m = masses[frag]
            com = (step_pos[frag] * m[:, None]).sum(axis=0) / m.sum()
            inter.append(step_delta[frag].sum(axis=0))
            torque.append(np.cross(step_pos[frag] - com, step_delta[frag]).sum(axis=0))
        _rigid, internal = rigid_body_split(step_delta, step_pos, fragment_atom_indices)
        intra.append(internal.ravel())

    return {
        "intermolecular": np.concatenate([np.ravel(v) for v in inter]),
        "torque": np.concatenate([np.ravel(v) for v in torque]),
        "intramolecular": np.concatenate(intra),
    }


FORCE_PART_KEYS = {
    "intermolecular": "rmsd_force_intermolecular",
    "torque": "rmsd_torque",
    "intramolecular": "rmsd_force_intramolecular",
}


def summarize_force_decomposition(per_system_parts: List[dict]) -> dict:
    """RMS of each part of :func:`decompose_force_errors_rigid`, pooled
    over systems. Returns ``{rmsd_force_intermolecular, rmsd_torque,
    rmsd_force_intramolecular}``; NaN for a part with no values."""
    out = {}
    for part, key in FORCE_PART_KEYS.items():
        chunks = [np.asarray(p[part], dtype=float).ravel() for p in per_system_parts if part in p]
        values = np.concatenate(chunks) if chunks else np.empty(0)
        out[key] = float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")
    return out


def combined_zscore_ranking(
    metrics_df: pd.DataFrame, weights: Dict[str, float]
) -> pd.DataFrame:
    """Combine per-method metrics into a weighted z-score ranking.

    The metrics combined are taken from ``weights.keys()`` — the same
    function therefore serves the 2-metric geom-benchmark case
    (``rmsd_eq`` + ``rmsd_force``) and any future N-metric variant
    (e.g. energy_benchmark) without modification.

    For each metric column, computes ``z = (x − mean) / std`` across the
    methods present in ``metrics_df`` (population std, ``ddof=0``).
    Lower combined score = better functional.

    Parameters
    ----------
    metrics_df : DataFrame
        Index = method name. Columns must include every key of
        ``weights``.
    weights : dict
        ``{metric_name → float}``. Determines which columns are combined
        and how heavily each is weighted.

    Returns
    -------
    DataFrame indexed by method, with z-score columns (named
    ``z_<metric>``) plus ``combined_score``, sorted ascending by
    ``combined_score``.

    Notes
    -----
    A metric column where all methods are equal (zero std) contributes
    ``0`` to every z-score — that dimension carries no ranking signal.
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    metric_cols = list(weights.keys())
    for col in metric_cols:
        if col not in metrics_df.columns:
            raise ValueError(f"metrics_df missing required column: {col}")

    out = pd.DataFrame(index=metrics_df.index)
    for col in metric_cols:
        x = metrics_df[col].astype(float)
        mean = float(x.mean())
        std = float(x.std(ddof=0))
        if std == 0 or np.isnan(std):
            out[f"z_{col}"] = 0.0
        else:
            out[f"z_{col}"] = (x - mean) / std

    out["combined_score"] = sum(
        float(weights[col]) * out[f"z_{col}"] for col in metric_cols
    )
    return out.sort_values("combined_score")
