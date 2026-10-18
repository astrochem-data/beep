"""Normal-mode displacement sampling — pure math.

Used by the ``nm_sampling`` workflow to:
  1. Run vibrational analysis on a Hessian record (wraps ``core.zpve``).
  2. Classify each mode as intermolecular / bending / stretching via
     rigid-body kinetic-energy projection (fragment-COM translation +
     rotation about COM).
  3. Pick the lowest-frequency modes from each band, up to a configured cap.
  4. Generate ± displaced Cartesian geometries at a target RMS amplitude.
  5. Write a Molden-format file + a programmatic JSON for visualization.

No QCFractal dependencies — all functions take plain numpy arrays plus a
qcelemental Molecule. The workflow layer is responsible for fetching the
Hessian record off the server and unpacking it into the array shape used here.
"""
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .zpve import _vibanal_wfn


# Bohr ↔ Å (CODATA 2018). qcelemental has these too but inlining keeps the
# module dependency-light + matches the constants used in trajectory_metrics.
_BOHR_TO_A = 0.529177210903


ModeBand = Literal["intermolecular", "bending", "stretching"]


# ---------------------------------------------------------------------------
# Hessian → normal modes (wrap zpve._vibanal_wfn)
# ---------------------------------------------------------------------------

def extract_normal_modes_from_hessian_record(
    hess: np.ndarray, molecule, energy: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run vibrational analysis on a Hessian.

    Parameters
    ----------
    hess : (3N, 3N) ndarray
        Non-mass-weighted Cartesian Hessian, units ``Eh / bohr²``. Shape
        matches the raw ``record.return_result`` of a hessian SP after
        reshape.
    molecule : qcelemental.models.Molecule
        Geometry + atomic masses container.
    energy : float, optional
        Electronic energy (Eh). Only used by ``_vibanal_wfn`` to populate
        the thermodynamic info dict; the modes themselves do not depend
        on it.

    Returns
    -------
    frequencies_cm : (n_vib,) complex ndarray
        Vibrational frequencies in cm⁻¹. Imaginary frequencies appear as
        purely imaginary complex values (real part 0); real frequencies
        appear with imaginary part 0. Translation/rotation modes are
        stripped from the array.
    modes_cart : (n_vib, n_atoms, 3) ndarray
        Cartesian (un-mass-weighted) displacement pattern per mode.
        Direction only — the absolute scale follows numpy's eigh
        normalisation and gets renormalised by ``displace_along_mode``.
    """
    vibinfo, _therminfo = _vibanal_wfn(hess=hess, molecule=molecule, energy=energy)
    trv = vibinfo["TRV"].data           # list of "TR" or "V" per mode
    omega = vibinfo["omega"].data       # (ndof,) complex
    modes = vibinfo["modes_cart"].data  # (ndof, n_atoms, 3)

    keep = [i for i, label in enumerate(trv) if label == "V"]
    if not keep:
        # All TR — shouldn't happen for a bound molecule but be safe.
        return np.empty(0, dtype=complex), np.empty((0, *modes.shape[1:]))
    return omega[keep], modes[keep]


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------

def _fragment_rigid_kinetic(
    positions: np.ndarray, masses: np.ndarray, mode_cart: np.ndarray,
) -> float:
    """Rigid-body kinetic energy (× 2) of one fragment in a normal mode.

    Returns ``K_trans + K_rot`` for the fragment, where

      * ``K_trans = M_f · |c_f|²``       (fragment-COM translation)
      * ``K_rot   = Lᵀ · I⁺ · L``        (rotation about fragment COM)

    ``c_f`` is the mass-weighted COM displacement, ``L`` the angular
    momentum about the COM, and ``I⁺`` the Moore–Penrose pseudo-inverse
    of the inertia tensor (which gracefully handles single-atom and
    collinear fragments — for those the null-space component of ``L``
    is projected out and contributes zero rotational KE).

    The constant ``½`` is dropped — only the ratio ``K_rigid / K_total``
    is used downstream, so the factor cancels.
    """
    if masses.size == 0:
        return 0.0
    M = float(masses.sum())
    if M <= 0.0:
        return 0.0
    com_pos = (masses[:, None] * positions).sum(axis=0) / M
    com_disp = (masses[:, None] * mode_cart).sum(axis=0) / M
    rho = positions - com_pos
    L = (masses[:, None] * np.cross(rho, mode_cart)).sum(axis=0)
    rho_sq = float((masses * (rho * rho).sum(axis=1)).sum())
    I = rho_sq * np.eye(3) - np.einsum("a,ai,aj->ij", masses, rho, rho)
    omega = np.linalg.pinv(I) @ L
    k_trans = M * float(com_disp @ com_disp)
    k_rot = float(L @ omega)
    return k_trans + k_rot


def classify_mode(
    mode_cart: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    fragment_atom_indices: Sequence[Sequence[int]],
    frequency_cm: float,
    inter_threshold: float = 0.5,
    bend_max_cm: float = 1500.0,
) -> ModeBand:
    """Classify a single normal mode.

    Uses a two-step rule:

      1. Compute ``f_inter`` — the fraction of the mode's kinetic energy
         carried by **rigid-body motion of each fragment** (translation
         of the fragment COM plus rotation about that COM), summed over
         every fragment. If ``f_inter > inter_threshold`` the mode is
         labelled *intermolecular* — it primarily moves whole fragments
         against one another, whether by translation or by libration.

      2. Otherwise the mode is intramolecular; split by frequency:
         ``frequency_cm < bend_max_cm`` → *bending*, else *stretching*.
         (Frequency is a robust separator for intramolecular modes;
         only for mixed-character modes does it fail, which is exactly
         what step 1 catches.)

    Including the rotational term is essential for low-frequency
    librations: a libration is a pure intramolecular rotation of one
    fragment about its own COM, so the COM *displacement* is ~0 even
    though every atom in the fragment is moving relative to the rest of
    the complex. A COM-displacement-only ratio misses librations and
    labels them as internal bends.

    Generalises transparently from adsorbate+cluster (N=2) to arbitrary
    N-mer systems: pass one atom-index list per fragment and the
    per-fragment rigid-body kinetic contributions are summed. Disjoint
    fragments have orthogonal rigid-body subspaces, so the sum is
    well-defined and bounded above by 1.

    Parameters
    ----------
    mode_cart : (n_atoms, 3) ndarray
        Cartesian displacement pattern of the mode (units irrelevant —
        only the relative atomic motion is used).
    masses : (n_atoms,) ndarray
        Atomic masses (amu).
    positions : (n_atoms, 3) ndarray
        Equilibrium atomic positions, same Cartesian frame as
        ``mode_cart``. Used to compute angular momenta + inertia
        tensors about each fragment COM. Units cancel in the ratio.
    fragment_atom_indices : list of list-of-int
        One 0-indexed atom-index list per fragment (matches
        ``qcel.Molecule.fragments`` shape). Every atom must appear in
        exactly one fragment. A single fragment covering the whole
        system reduces f_inter to 0 and lets the frequency cut decide.
    frequency_cm : float
        Mode frequency (real part if complex) in cm⁻¹.
    inter_threshold : float, default 0.5
        ``f_inter`` above this → intermolecular.
    bend_max_cm : float, default 1500.0
        Intramolecular cutoff between bending and stretching.
    """
    n_frags = len(fragment_atom_indices) if fragment_atom_indices else 0
    if n_frags <= 1:
        # Degenerate fragmenting (whole system as one fragment or empty) —
        # rigid-body motion is a null operation on the whole system's
        # vibrational subspace, so f_inter is 0 and the frequency cut decides.
        f_inter = 0.0
    else:
        k_rigid = 0.0
        for frag_idx in fragment_atom_indices:
            idx = np.asarray(frag_idx, dtype=int)
            if idx.size == 0:
                continue
            k_rigid += _fragment_rigid_kinetic(
                positions[idx], masses[idx], mode_cart[idx],
            )
        k_total = float(np.einsum("a,ai,ai->", masses, mode_cart, mode_cart))
        f_inter = k_rigid / k_total if k_total > 0.0 else 0.0

    if f_inter > inter_threshold:
        return "intermolecular"
    if frequency_cm < bend_max_cm:
        return "bending"
    return "stretching"


# ---------------------------------------------------------------------------
# Per-band selection
# ---------------------------------------------------------------------------

def select_modes(
    frequencies_cm: np.ndarray,
    classes: List[str],
    band_caps: Dict[str, int],
    band_amplitudes: Dict[str, float],
    freq_max_imag_cm: float = 50.0,
    extra_amplitudes_lowest_count: int = 1,
    extra_amplitude_factor: float = 2.0,
) -> List[Tuple[int, float, str]]:
    """Pick which modes to displace and at what amplitude.

    Per band, take the ``cap`` lowest-frequency modes whose imaginary
    component is below the threshold. The N lowest-frequency entries of
    the final picked set (across all bands) get a second amplitude
    appended at ``amplitude × extra_amplitude_factor``.

    Parameters
    ----------
    frequencies_cm : (n_modes,) complex ndarray
        Mode frequencies. Real frequencies have ``imag == 0``; imaginary
        frequencies have ``real == 0``.
    classes : list of str, length n_modes
        Each element is "intermolecular", "bending", or "stretching".
    band_caps, band_amplitudes : dict
        Per-band cap (int) and amplitude (Å, RMS Cartesian).
    freq_max_imag_cm : float, default 50.0
        Drop modes whose |imag| exceeds this magnitude.
    extra_amplitudes_lowest_count : int, default 1
        Number of lowest-frequency selected modes that get the extra
        amplitude.
    extra_amplitude_factor : float, default 2.0
        Multiplier for the extra-amplitude entries.

    Returns
    -------
    list of (mode_index, amplitude_A, band_label)
        Each entry represents a displacement pair (the caller submits
        both + and − signs separately). Sorted ascending by frequency
        within bands; extra-amplitude entries trail their primary entry.
    """
    selected: List[Tuple[int, float, str]] = []

    # Build per-band candidate lists (mode_index, real_freq), filtered.
    candidates: Dict[str, List[Tuple[int, float]]] = {b: [] for b in band_caps}
    for idx, (omega, band) in enumerate(zip(frequencies_cm, classes)):
        if band not in candidates:
            continue
        re = float(np.real(omega))
        im = float(np.imag(omega))
        if abs(im) > freq_max_imag_cm:
            continue
        candidates[band].append((idx, re))

    # Sort each band's candidates by ascending real frequency, take the cap.
    for band, cap in band_caps.items():
        picks = sorted(candidates[band], key=lambda x: x[1])[: int(cap)]
        amp = float(band_amplitudes[band])
        for idx, _freq in picks:
            selected.append((idx, amp, band))

    # Identify the N lowest-frequency entries across the whole selection
    # and append extra-amplitude duplicates.
    if extra_amplitudes_lowest_count > 0 and selected:
        # Pair each entry with its frequency for ranking.
        ranked = sorted(
            range(len(selected)),
            key=lambda i: float(np.real(frequencies_cm[selected[i][0]])),
        )
        extras = []
        for slot in ranked[: int(extra_amplitudes_lowest_count)]:
            idx, amp, band = selected[slot]
            extras.append((idx, amp * float(extra_amplitude_factor), band))
        selected.extend(extras)

    return selected


# ---------------------------------------------------------------------------
# Displacement geometry generation
# ---------------------------------------------------------------------------

def displace_along_mode(
    geometry_bohr: np.ndarray,
    mode_cart: np.ndarray,
    amplitude_A: float,
    sign: int = +1,
) -> np.ndarray:
    """Apply a ±-signed displacement along a mode to a geometry.

    The mode is scaled so its RMS Cartesian length (over all atoms) equals
    ``amplitude_A`` (in Å), converted to bohr, then added to the
    equilibrium geometry with the given sign.

    Parameters
    ----------
    geometry_bohr : (n_atoms, 3) ndarray
        Equilibrium Cartesian geometry in bohr.
    mode_cart : (n_atoms, 3) ndarray
        Mode direction in Cartesians (un-mass-weighted). Magnitude is
        irrelevant — only the direction is used.
    amplitude_A : float
        Target RMS Cartesian displacement of all atoms, in Å.
    sign : int, default +1
        ``+1`` or ``-1``.

    Returns
    -------
    (n_atoms, 3) ndarray — displaced geometry in bohr.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    n_atoms = mode_cart.shape[0]
    if n_atoms == 0:
        return geometry_bohr.copy()

    # RMS Cartesian length of the mode vector, in whatever units the input
    # came with. We rescale to amplitude_A (in bohr) — RMS of a target
    # displacement field of RMS=A_bohr has Σ|v|² = 3 N A_bohr² (each of
    # 3 N components contributes A_bohr²/(3N) on average, and the RMS of
    # a vector with 3 N components is √(Σ|v_i|² / (3N))).
    sum_sq = float(np.einsum("ai,ai->", mode_cart, mode_cart))
    if sum_sq <= 0.0:
        return geometry_bohr.copy()
    rms_mode = np.sqrt(sum_sq / (3.0 * n_atoms))
    amp_bohr = amplitude_A / _BOHR_TO_A
    scale = amp_bohr / rms_mode

    return geometry_bohr + sign * scale * mode_cart


# ---------------------------------------------------------------------------
# Molden + JSON writers (visualization / programmatic post-analysis)
# ---------------------------------------------------------------------------

def write_molden(
    filepath,
    symbols: List[str],
    geometry_bohr: np.ndarray,
    frequencies_cm: np.ndarray,
    modes_cart: np.ndarray,
    title: str = "BEEP nm_sampling normal modes",
) -> None:
    """Write Molden-format normal-mode file (visualisable in Avogadro, jmol, IboView).

    Imaginary frequencies are written with a negative sign (Molden's
    convention for saddle-direction modes).

    Parameters
    ----------
    filepath : str or Path
        Output ``.molden`` path.
    symbols : list of str, length n_atoms
        Atomic symbols.
    geometry_bohr : (n_atoms, 3) ndarray
        Atomic positions in Bohr.
    frequencies_cm : (n_vib,) complex ndarray
        Vibrational frequencies; real modes have ``imag == 0``, imaginary
        modes have ``real == 0`` (qcelemental convention).
    modes_cart : (n_vib, n_atoms, 3) ndarray
        Cartesian displacement patterns (un-mass-weighted).
    title : str
        Title line written into the ``[Title]`` block.
    """
    import qcelemental as qcel
    atomic_numbers = [qcel.periodictable.to_atomic_number(s) for s in symbols]
    n_atoms = len(symbols)

    lines = ["[Molden Format]", "[Title]", title, "[Atoms] AU"]
    for i, (sym, Z) in enumerate(zip(symbols, atomic_numbers), start=1):
        x, y, z = geometry_bohr[i - 1]
        lines.append(
            f"{sym:<3s} {i:>5d} {Z:>4d}  "
            f"{x:>16.10f} {y:>16.10f} {z:>16.10f}"
        )

    lines.append("[FREQ]")
    for omega in frequencies_cm:
        re = float(np.real(omega))
        im = float(np.imag(omega))
        value = -abs(im) if abs(im) > abs(re) else re
        lines.append(f"{value:>14.4f}")

    lines.append("[FR-COORD]")
    for sym, (x, y, z) in zip(symbols, geometry_bohr):
        lines.append(f"{sym:<3s} {x:>16.10f} {y:>16.10f} {z:>16.10f}")

    lines.append("[FR-NORM-COORD]")
    for i, mode in enumerate(modes_cart, start=1):
        lines.append(f"vibration {i}")
        for dx, dy, dz in mode:
            lines.append(f"  {dx:>14.10f} {dy:>14.10f} {dz:>14.10f}")

    Path(filepath).write_text("\n".join(lines) + "\n")


def write_modes_json(
    filepath,
    symbols: List[str],
    geometry_bohr: np.ndarray,
    frequencies_cm: np.ndarray,
    modes_cart: np.ndarray,
    classes: List[str],
    fragment_atom_indices: Sequence[Sequence[int]],
    level_of_theory: str = "",
) -> None:
    """Write a per-system normal-mode JSON.

    Structure: symbols, geometry in Å, the N-fragment partition
    (``fragments`` key — one 0-indexed atom-index list per fragment,
    matching ``qcel.Molecule.fragments`` shape), and a list of
    ``{freq_cm, disp, class}`` entries per mode.

    Parameters
    ----------
    filepath : str or Path
        Output ``.json`` path.
    symbols : list of str, length n_atoms
    geometry_bohr : (n_atoms, 3) ndarray
        Positions in Bohr — converted to Å on write.
    frequencies_cm : (n_vib,) complex ndarray
    modes_cart : (n_vib, n_atoms, 3) ndarray
    classes : list of str, length n_vib
        Output of ``classify_mode`` per mode.
    fragment_atom_indices : list of list-of-int
        Same partition used for classification.
    level_of_theory : str
        For the ``level_of_theory`` key — e.g. ``hf_def2-svp``.
    """
    geom_A = (np.asarray(geometry_bohr) * _BOHR_TO_A).tolist()
    fragments_json = [list(int(i) for i in frag) for frag in fragment_atom_indices]

    modes = []
    for omega, disp, cls in zip(frequencies_cm, modes_cart, classes):
        re = float(np.real(omega))
        im = float(np.imag(omega))
        freq = re if abs(re) > abs(im) else im  # store imag with positive magnitude
        modes.append({
            "freq_cm": freq,
            "is_imaginary": bool(abs(im) > abs(re)),
            "class": cls,
            "disp": np.asarray(disp).tolist(),
        })

    payload = {
        "level_of_theory": level_of_theory,
        "symbols": list(symbols),
        "geometry_A": geom_A,
        "fragments": fragments_json,
        "modes": modes,
    }
    Path(filepath).write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Force-error decomposition (offset / response / rigid-body)
# ---------------------------------------------------------------------------

def decompose_force_errors(
    samples: Sequence[dict],
    fragment_atom_indices: Sequence[Sequence[int]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Split per-geometry force errors of one system into physical parts.

    Each sample is one displaced geometry::

        {"pair": (mode_idx, amplitude_A), "sign": +1 | -1, "band": str,
         "delta": (n_atoms, 3) DFT − reference force error, meV/Å,
         "positions_A": (n_atoms, 3), "masses": (n_atoms,)}

    Every mode is displaced by +A and −A around the same centre, so the
    error of a ± pair splits exactly into

    * ``offset``    = (δ₊ + δ₋)/2 — the error already present at the
      centre (e.g. a functional whose bond lengths differ from the
      reference feels a restoring force at every geometry);
    * ``response``  = (δ₊ − δ₋)/2 — the force error that grows with the
      displacement amplitude (it originates in a second derivative that
      differs from the reference); the offset cancels exactly.

    The rigid-body parts are taken per fragment at every geometry:

    * ``intermolecular`` — net force error on each fragment (Σ δ_i). Internal
      forces cancel in the sum, so this is purely intermolecular;
    * ``torque`` — torque error about each fragment's centre of mass,
      Σ (r_i − r_COM) × δ_i, in meV; also purely intermolecular.

    Returns ``{part → {band → flat array}}`` with an ``"all"`` band holding
    every value. Pairs with a missing partner are skipped for
    offset/response but still contribute to the rigid-body parts.
    """
    parts = ("offset", "response", "intermolecular", "torque")
    out: Dict[str, Dict[str, list]] = {p: {"all": []} for p in parts}

    def _add(part, band, values):
        values = np.asarray(values, dtype=float).ravel()
        out[part]["all"].append(values)
        out[part].setdefault(band, []).append(values)

    by_pair: Dict[tuple, Dict[int, dict]] = {}
    for s in samples:
        by_pair.setdefault(tuple(s["pair"]), {})[int(s["sign"])] = s
        delta = np.asarray(s["delta"], dtype=float)
        pos = np.asarray(s["positions_A"], dtype=float)
        masses = np.asarray(s["masses"], dtype=float)
        for frag in fragment_atom_indices:
            frag = list(frag)
            m = masses[frag]
            com = (pos[frag] * m[:, None]).sum(axis=0) / m.sum()
            _add("intermolecular", s["band"], delta[frag].sum(axis=0))
            _add("torque", s["band"],
                 np.cross(pos[frag] - com, delta[frag]).sum(axis=0))

    for pair in by_pair.values():
        if 1 not in pair or -1 not in pair:
            continue
        plus, minus = pair[1], pair[-1]
        d_plus = np.asarray(plus["delta"], dtype=float)
        d_minus = np.asarray(minus["delta"], dtype=float)
        _add("offset", plus["band"], (d_plus + d_minus) / 2)
        _add("response", plus["band"], (d_plus - d_minus) / 2)

    return {
        part: {
            band: np.concatenate(chunks) if chunks else np.empty(0)
            for band, chunks in bands.items()
        }
        for part, bands in out.items()
    }


# ---------------------------------------------------------------------------
# Force offset -> predicted geometry shift (Newton step)
# ---------------------------------------------------------------------------

def _sqrt_au_to_cm() -> float:
    """cm⁻¹ per sqrt(Eh / (bohr² · amu)) — frequency of a mass-weighted
    Hessian eigenvalue of 1 in atomic units."""
    import qcelemental as qcel
    c = qcel.constants
    eh = c.conversion_factor("hartree", "J")
    bohr = c.conversion_factor("bohr", "m")
    amu = c.conversion_factor("amu", "kg")
    c_cm = c.c * 100.0
    return float(np.sqrt(eh / (bohr ** 2 * amu)) / (2.0 * np.pi * c_cm))


def hessian_pseudo_inverse(
    hessian: np.ndarray, masses_amu: np.ndarray, positions_bohr: np.ndarray,
    min_freq_cm: float = 20.0,
) -> Tuple[np.ndarray, int]:
    """Pseudo-inverse of a Cartesian Hessian restricted to real vibrations.

    Translations and rotations are projected out in mass-weighted
    coordinates; modes with frequency below ``min_freq_cm`` (near-free
    motions, where 1/k amplifies noise) and imaginary modes are dropped.

    Returns ``(H_plus, n_modes_kept)`` with ``H_plus`` in bohr²/Eh, so that
    ``H_plus @ gradient_Eh_per_bohr`` is a displacement in bohr.
    """
    m = np.repeat(np.asarray(masses_amu, dtype=float), 3)
    inv_sqrt_m = 1.0 / np.sqrt(m)
    hmw = inv_sqrt_m[:, None] * np.asarray(hessian, dtype=float) * inv_sqrt_m[None, :]

    pos = np.asarray(positions_bohr, dtype=float).reshape(-1, 3)
    mass = np.asarray(masses_amu, dtype=float)
    com = (pos * mass[:, None]).sum(axis=0) / mass.sum()
    r = pos - com
    sqrt_m_atom = np.sqrt(mass)
    tr = []
    for axis in range(3):
        v = np.zeros_like(pos); v[:, axis] = sqrt_m_atom
        tr.append(v.ravel())
        e = np.zeros(3); e[axis] = 1.0
        tr.append((np.cross(e, r) * sqrt_m_atom[:, None]).ravel())
    # Orthonormal basis of the vibrational subspace (complement of the
    # translations/rotations). Diagonalising only there keeps the TR
    # directions out exactly, instead of relying on near-zero eigenvalues.
    u, sv, _ = np.linalg.svd(np.array(tr).T, full_matrices=True)
    n_tr = int(np.sum(sv > 1e-8 * sv.max()))
    vib = u[:, n_tr:]
    eigval, eigvec_vib = np.linalg.eigh(vib.T @ hmw @ vib)
    eigvec = vib @ eigvec_vib

    freqs = np.sign(eigval) * np.sqrt(np.abs(eigval)) * _sqrt_au_to_cm()
    keep = (eigval > 0) & (freqs >= min_freq_cm)
    pinv_mw = (eigvec[:, keep] / eigval[keep]) @ eigvec[:, keep].T
    h_plus = inv_sqrt_m[:, None] * pinv_mw * inv_sqrt_m[None, :]
    return h_plus, int(keep.sum())


def mean_pair_offset(samples: Sequence[dict]) -> Optional[np.ndarray]:
    """Average (δ₊ + δ₋)/2 over every complete ± pair of one system.

    Every pair is centred on the same geometry, so each pair estimates the
    same force error at the centre; averaging reduces the O(A²) noise.
    Returns an ``(n_atoms, 3)`` array in the units of ``delta``, or None.
    """
    by_pair: Dict[tuple, Dict[int, np.ndarray]] = {}
    for s in samples:
        by_pair.setdefault(tuple(s["pair"]), {})[int(s["sign"])] = np.asarray(
            s["delta"], dtype=float,
        )
    offsets = [
        (p[1] + p[-1]) / 2 for p in by_pair.values() if 1 in p and -1 in p
    ]
    return np.mean(offsets, axis=0) if offsets else None


def rigid_body_split(
    displacement: np.ndarray, positions: np.ndarray,
    fragment_atom_indices: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a small displacement into per-fragment rigid motion
    (translation + rotation, least squares) and the internal remainder."""
    dx = np.asarray(displacement, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    rigid = np.zeros_like(dx)
    for frag in fragment_atom_indices:
        frag = list(frag)
        r = pos[frag] - pos[frag].mean(axis=0)
        a = np.zeros((3 * len(frag), 6))
        for k, ri in enumerate(r):
            a[3 * k:3 * k + 3, :3] = np.eye(3)
            a[3 * k:3 * k + 3, 3:] = -np.array([
                [0.0, -ri[2], ri[1]], [ri[2], 0.0, -ri[0]], [-ri[1], ri[0], 0.0],
            ])
        sol, *_ = np.linalg.lstsq(a, dx[frag].ravel(), rcond=None)
        rigid[frag] = (a @ sol).reshape(-1, 3)
    return rigid, dx - rigid


def predicted_geometry_shift(
    offset_gradient: np.ndarray, hessian_plus: np.ndarray,
    positions_bohr: np.ndarray, fragment_atom_indices: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """First-order shift of the functional's minimum from the reference one.

    ``Δx = H⁺ · Δg`` with ``Δg`` the gradient error at the displacement
    centre (Eh/bohr). Reported as an RMS over atoms in Å, total and split
    into the intermolecular (rigid per fragment) and intramolecular parts.
    """
    bohr_to_A = 0.529177210903
    dx = (hessian_plus @ np.asarray(offset_gradient, dtype=float).ravel())
    dx = dx.reshape(-1, 3) * bohr_to_A
    rigid, internal = rigid_body_split(
        dx, np.asarray(positions_bohr).reshape(-1, 3) * bohr_to_A,
        fragment_atom_indices,
    )
    rms = lambda v: float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))
    return {
        "geometry_shift": rms(dx),
        "geometry_shift_intermolecular": rms(rigid),
        "geometry_shift_intramolecular": rms(internal),
    }
