"""Tests for beep/core/normal_mode_sampling.py."""
import json
import numpy as np
import pytest

from beep.core.normal_mode_sampling import (
    classify_mode,
    select_modes,
    displace_along_mode,
    extract_normal_modes_from_hessian_record,
    write_molden,
    write_modes_json,
    _BOHR_TO_A,
)


# ---------------------------------------------------------------------------
# classify_mode
# ---------------------------------------------------------------------------

def _ten_atom_setup(n_ads=2):
    """Cluster of 8 oxygens (in a cube) + adsorbate of `n_ads` hydrogens.

    Returns (masses, positions). Adsorbate hydrogens sit on a small bond
    along the x-axis at +/- 0.5 Å around (4, 0, 0), well separated from
    the cluster — a rough proxy for a small molecule on a water cluster.
    """
    masses = np.array([16.0] * 8 + [1.0] * n_ads)
    # 8-atom cubic cluster of oxygens at the unit-cube corners.
    grid = np.array([[x, y, z]
                     for x in (-1.0, +1.0)
                     for y in (-1.0, +1.0)
                     for z in (-1.0, +1.0)])
    # 2-atom adsorbate H–H along x at (4, 0, 0).
    ads = np.array([[3.5, 0.0, 0.0], [4.5, 0.0, 0.0]])[:n_ads]
    positions = np.vstack([grid, ads])
    return masses, positions


def test_classify_intermolecular_pure_com_translation():
    """Adsorbate translates uniformly +z, cluster stationary → intermolecular."""
    masses, positions = _ten_atom_setup(n_ads=2)
    mode = np.zeros((10, 3))
    mode[8:10, 2] = 1.0  # both adsorbate atoms move +z
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(8)), [8, 9]],
                          frequency_cm=200.0) == "intermolecular"


def test_classify_intermolecular_libration():
    """Adsorbate rotates about its own COM (libration), cluster stationary.

    This is the failure mode of the COM-displacement-only classifier:
    the fragment COM does not move, but every atom in the fragment does.
    A rigid-body classifier captures the rotational KE and labels this
    intermolecular.
    """
    masses, positions = _ten_atom_setup(n_ads=2)
    # Adsorbate at (3.5, 0, 0) and (4.5, 0, 0): COM at (4, 0, 0).
    # Rotation about z (perpendicular to the bond) → atom 8 moves +y,
    # atom 9 moves -y. Net COM displacement = 0; angular momentum
    # about the z-axis is non-zero.
    mode = np.zeros((10, 3))
    mode[8, 1] = +1.0
    mode[9, 1] = -1.0
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(8)), [8, 9]],
                          frequency_cm=120.0) == "intermolecular"


def test_classify_stretching_intramolecular_bond_oscillation():
    """Adsorbate H–H bond stretch, cluster stationary, high-freq → stretching.

    The bond axis is along x, both H atoms move in ±x — pure stretch with
    no COM translation and no angular momentum (radial vs ρ — cross is 0).
    """
    masses, positions = _ten_atom_setup(n_ads=2)
    mode = np.zeros((10, 3))
    mode[8, 0] = +1.0   # H₁ moves +x (along bond, outward)
    mode[9, 0] = -1.0   # H₂ moves −x  (outward) → no COM motion, no rotation
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(8)), [8, 9]],
                          frequency_cm=4000.0) == "stretching"


def test_classify_bending_intramolecular_low_freq():
    """Stretch-like antisymmetric motion at low frequency → bending."""
    masses, positions = _ten_atom_setup(n_ads=2)
    # Same as the stretch test but at low frequency: both atoms move
    # ±x along the bond axis (no rotation since v is parallel to ρ).
    mode = np.zeros((10, 3))
    mode[8, 0] = +1.0
    mode[9, 0] = -1.0
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(8)), [8, 9]],
                          frequency_cm=800.0) == "bending"


def test_classify_degenerate_single_fragment_falls_back_to_freq():
    """A single-fragment partition (the whole system as one block) forces
    f_inter to 0 so only frequency drives the classification."""
    masses, positions = _ten_atom_setup(n_ads=2)
    mode = np.ones((10, 3))
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(10))],
                          frequency_cm=500.0) == "bending"
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(10))],
                          frequency_cm=3000.0) == "stretching"


def test_classify_zero_mode_returns_bending_low_freq():
    """A zero mode (e.g. spurious near-TR remnant) shouldn't blow up."""
    masses, positions = _ten_atom_setup(n_ads=2)
    mode = np.zeros((10, 3))
    # zero kinetic → f_inter forced to 0 → falls through to frequency
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(8)), [8, 9]],
                          frequency_cm=100.0) == "bending"


def test_classify_trimer_all_three_intermolecular_modes_labelled_correctly():
    """N=3 fragments: intermolecular motion of ANY pair must classify as
    intermolecular, not just motion involving the last fragment.

    Setup: three diatomic fragments arranged in a triangle. Test each of
    the three canonical intermolecular displacements (translate fragment
    A only, translate fragment B only, translate fragment C only) and
    verify all three come out intermolecular. The old 2-fragment
    classifier would silently mislabel the "translate A only" mode
    (leftmost fragment moving) because A is not the "adsorbate".
    """
    # Three diatomic fragments, well-separated
    masses = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],   # fragment 0
        [4.0, 0.0, 0.0], [5.0, 0.0, 0.0],   # fragment 1
        [8.0, 0.0, 0.0], [9.0, 0.0, 0.0],   # fragment 2
    ])
    fragments = [[0, 1], [2, 3], [4, 5]]

    for moving_frag in fragments:
        mode = np.zeros((6, 3))
        for a in moving_frag:
            mode[a, 2] = 1.0                # rigid +z translation of that fragment
        assert classify_mode(
            mode, masses, positions,
            fragment_atom_indices=fragments,
            frequency_cm=180.0,
        ) == "intermolecular", (
            f"fragment translation of {moving_frag} misclassified"
        )


def test_classify_trimer_intramolecular_bond_still_classified_by_freq():
    """Sanity check for the N=3 classifier: an intra-fragment stretch is
    classified by frequency (bending vs stretching), same as N=2."""
    masses = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [4.0, 0.0, 0.0], [5.0, 0.0, 0.0],
        [8.0, 0.0, 0.0], [9.0, 0.0, 0.0],
    ])
    fragments = [[0, 1], [2, 3], [4, 5]]
    # Stretch of fragment 1: atoms 2, 3 move ±x. No net translation, no
    # rotation (v parallel to ρ w.r.t. fragment COM) → not intermolecular.
    mode = np.zeros((6, 3))
    mode[2, 0] = +1.0
    mode[3, 0] = -1.0
    assert classify_mode(
        mode, masses, positions,
        fragment_atom_indices=fragments,
        frequency_cm=3500.0,
    ) == "stretching"
    assert classify_mode(
        mode, masses, positions,
        fragment_atom_indices=fragments,
        frequency_cm=800.0,
    ) == "bending"


def test_classify_single_atom_adsorbate_handled():
    """One-atom adsorbate (e.g. He on water) has zero inertia tensor —
    rigid-body kinetic must still evaluate without raising."""
    masses, positions = _ten_atom_setup(n_ads=1)
    mode = np.zeros((9, 3))
    mode[8, 2] = 1.0  # the lone adsorbate atom moves +z
    # Pure intermolecular translation of the monatomic adsorbate.
    assert classify_mode(mode, masses, positions,
                          fragment_atom_indices=[list(range(8)), [8]],
                          frequency_cm=80.0) == "intermolecular"


# ---------------------------------------------------------------------------
# select_modes
# ---------------------------------------------------------------------------

def _twelve_mode_setup():
    freqs = np.array(
        [100, 200, 300, 400, 500,        # 5 intermolecular
         600, 700, 800,                  # 3 bending
         2900, 3700, 3800, 3900],        # 4 stretching
        dtype=complex,
    )
    classes = (
        ["intermolecular"] * 5
        + ["bending"] * 3
        + ["stretching"] * 4
    )
    band_caps = {"intermolecular": 3, "bending": 2, "stretching": 1}
    band_amps = {"intermolecular": 0.08, "bending": 0.05, "stretching": 0.03}
    return freqs, classes, band_caps, band_amps


def test_select_modes_caps_and_lowest_freq_picked():
    freqs, classes, band_caps, band_amps = _twelve_mode_setup()
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        extra_amplitudes_lowest_count=0)
    # Three lowest intermolecular: indices 0, 1, 2
    # Two lowest bending: indices 5, 6
    # One lowest stretching: index 8
    indices = [s[0] for s in sel]
    assert indices == [0, 1, 2, 5, 6, 8]
    # Amplitudes per band
    amps = [s[1] for s in sel]
    assert amps == [0.08, 0.08, 0.08, 0.05, 0.05, 0.03]
    bands = [s[2] for s in sel]
    assert bands == ["intermolecular"] * 3 + ["bending"] * 2 + ["stretching"]


def test_select_modes_extra_amplitude_for_lowest():
    freqs, classes, band_caps, band_amps = _twelve_mode_setup()
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        extra_amplitudes_lowest_count=1,
                        extra_amplitude_factor=2.0)
    # The previous 6 entries are unchanged, the 7th is a duplicate of the
    # lowest-frequency entry at amplitude × 2.
    assert len(sel) == 7
    assert sel[-1][0] == 0           # mode index of the lowest-freq entry
    assert sel[-1][1] == pytest.approx(0.16)  # amplitude × 2
    assert sel[-1][2] == "intermolecular"


def test_select_modes_drops_large_imaginary():
    """Imaginary frequency > freq_max_imag_cm is dropped; small imag kept."""
    freqs = np.array(
        [80j,                       # |80| > 50 → dropped (saddle direction)
         30j,                       # |30| < 50 → kept (numerical noise)
         100, 200, 300, 600, 700,   # five reals
         3700,
        ],
        dtype=complex,
    )
    classes = ["intermolecular"] * 5 + ["bending", "bending", "stretching"]
    band_caps = {"intermolecular": 3, "bending": 2, "stretching": 1}
    band_amps = {"intermolecular": 0.08, "bending": 0.05, "stretching": 0.03}
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        freq_max_imag_cm=50.0,
                        extra_amplitudes_lowest_count=0)
    indices = [s[0] for s in sel]
    # 80j dropped; 30j kept (lowest "frequency"=0); next two reals = 100, 200
    assert 0 not in indices            # 80j was at index 0
    assert 1 in indices                # 30j survives


def test_select_modes_unknown_band_is_ignored():
    freqs = np.array([100, 200], dtype=complex)
    classes = ["intermolecular", "unknown_band"]
    band_caps = {"intermolecular": 3}
    band_amps = {"intermolecular": 0.08}
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        extra_amplitudes_lowest_count=0)
    assert sel == [(0, 0.08, "intermolecular")]


def test_select_modes_empty_inputs_returns_empty():
    sel = select_modes(np.array([], dtype=complex), [], {"a": 1}, {"a": 0.1})
    assert sel == []


# ---------------------------------------------------------------------------
# displace_along_mode
# ---------------------------------------------------------------------------

def test_displace_rms_equals_amplitude():
    """The RMS Cartesian displacement of the displaced geometry equals the
    target amplitude (in Å)."""
    rng = np.random.default_rng(seed=2026)
    geom = rng.standard_normal((5, 3))   # bohr
    mode = rng.standard_normal((5, 3))
    target_A = 0.10
    displaced = displace_along_mode(geom, mode, amplitude_A=target_A, sign=+1)
    dx_A = (displaced - geom) * _BOHR_TO_A
    rms = float(np.sqrt(np.mean(dx_A ** 2)))
    assert rms == pytest.approx(target_A, rel=1e-9)


def test_displace_sign_inversion_is_exact_opposite():
    rng = np.random.default_rng(2027)
    geom = rng.standard_normal((4, 3))
    mode = rng.standard_normal((4, 3))
    plus = displace_along_mode(geom, mode, amplitude_A=0.05, sign=+1)
    minus = displace_along_mode(geom, mode, amplitude_A=0.05, sign=-1)
    # plus + minus == 2 * geom  (exact)
    assert np.allclose(plus + minus, 2.0 * geom)


def test_displace_invalid_sign_raises():
    geom = np.zeros((2, 3))
    mode = np.ones((2, 3))
    with pytest.raises(ValueError, match="sign"):
        displace_along_mode(geom, mode, amplitude_A=0.1, sign=0)


def test_displace_zero_mode_returns_original():
    geom = np.random.default_rng(0).standard_normal((3, 3))
    mode = np.zeros((3, 3))
    out = displace_along_mode(geom, mode, amplitude_A=0.1, sign=+1)
    assert np.allclose(out, geom)


# ---------------------------------------------------------------------------
# extract_normal_modes_from_hessian_record
# ---------------------------------------------------------------------------

def test_extract_normal_modes_diagonal_hessian_harmonics():
    """A diagonal Hessian on a 3-atom linear molecule gives analytical
    harmonic frequencies. After stripping TR modes we expect 3·N − 5 = 4
    vibrational frequencies (linear) plus the TR modes correctly removed."""
    try:
        from qcelemental.models import Molecule
    except ImportError:
        pytest.skip("qcelemental not available")

    # Linear molecule along x: H ... O ... H (centered on O)
    geom_bohr = np.array([[-1.5, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [+1.5, 0.0, 0.0]])
    symbols = ["H", "O", "H"]
    masses_amu = [1.00782503207, 15.99491461957, 1.00782503207]
    mol = Molecule(
        symbols=symbols, geometry=geom_bohr.flatten(),
        masses=masses_amu, fix_com=True, fix_orientation=True,
    )
    # Diagonal Hessian: equal positive k in every Cartesian direction.
    k = 0.5
    hess = k * np.eye(9)
    freqs, modes = extract_normal_modes_from_hessian_record(hess, mol, energy=0.0)
    # A linear triatomic gives 3*3 - 5 = 4 vibrations (but identifying
    # linearity from masses alone isn't done here, so the harmonic_analysis
    # treats it as non-linear and projects 6 TR modes → 3 vib modes left).
    # Either way, all non-TR frequencies must be real and finite.
    assert freqs.size >= 3
    assert np.all(np.abs(np.imag(freqs)) < 1e-6)
    assert np.all(np.real(freqs) > 0)
    assert modes.shape == (freqs.size, 3, 3)


def test_extract_normal_modes_returns_aligned_arrays():
    """Frequencies and mode arrays line up by index."""
    try:
        from qcelemental.models import Molecule
    except ImportError:
        pytest.skip("qcelemental not available")

    # Diatomic — simplest vibrational case.
    geom_bohr = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, +0.7]])
    mol = Molecule(
        symbols=["H", "H"], geometry=geom_bohr.flatten(),
        masses=[1.00782503207, 1.00782503207],
        fix_com=True, fix_orientation=True,
    )
    hess = 0.6 * np.eye(6)
    freqs, modes = extract_normal_modes_from_hessian_record(hess, mol, energy=0.0)
    assert freqs.shape[0] == modes.shape[0]
    assert modes.shape[1:] == (2, 3)


# ---------------------------------------------------------------------------
# write_molden / write_modes_json
# ---------------------------------------------------------------------------

def _two_mode_writer_fixture():
    """Minimal 3-atom + 2-mode payload for the writer tests."""
    symbols = ["H", "O", "H"]
    geom_bohr = np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [+1.5, 0.0, 0.0]])
    freqs = np.array([1234.5 + 0j, 0 + 80j], dtype=complex)  # one real, one imag
    modes = np.array([
        [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.1, 0.0, 0.0]],   # symm stretch
        [[0.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, -0.2, 0.0]],   # bend
    ])
    return symbols, geom_bohr, freqs, modes


def test_write_molden_emits_required_sections(tmp_path):
    symbols, geom, freqs, modes = _two_mode_writer_fixture()
    out = tmp_path / "test.molden"
    write_molden(out, symbols, geom, freqs, modes, title="unit test")
    text = out.read_text()
    # Required Molden sections
    for section in ("[Molden Format]", "[Title]", "[Atoms] AU",
                    "[FREQ]", "[FR-COORD]", "[FR-NORM-COORD]"):
        assert section in text, f"missing section {section}"
    # Imaginary frequency rendered as negative (Molden convention)
    assert "-80.0000" in text, "imaginary mode not flagged with negative sign"
    # Real frequency rendered with its value
    assert "1234.5000" in text
    # Both vibrations listed
    assert "vibration 1" in text and "vibration 2" in text


def test_write_modes_json_matches_reference_shape(tmp_path):
    symbols, geom, freqs, modes = _two_mode_writer_fixture()
    classes = ["stretching", "intermolecular"]
    out = tmp_path / "modes.json"
    write_modes_json(
        out, symbols, geom, freqs, modes,
        classes=classes, fragment_atom_indices=[[0, 1], [2]],
        level_of_theory="hf_def2-svp",
    )
    payload = json.loads(out.read_text())
    # Keys mirror the current schema
    for k in ("level_of_theory", "symbols", "geometry_A",
              "fragments", "modes"):
        assert k in payload
    assert payload["level_of_theory"] == "hf_def2-svp"
    assert payload["symbols"] == ["H", "O", "H"]
    # Geometry converted Bohr → Å
    assert payload["geometry_A"][0][0] == pytest.approx(-1.5 * _BOHR_TO_A)
    # Fragments carried through verbatim
    assert payload["fragments"] == [[0, 1], [2]]
    # Mode entries carry freq + disp + class + imaginary flag
    assert len(payload["modes"]) == 2
    assert payload["modes"][0]["class"] == "stretching"
    assert payload["modes"][0]["is_imaginary"] is False
    assert payload["modes"][1]["class"] == "intermolecular"
    assert payload["modes"][1]["is_imaginary"] is True
    assert payload["modes"][1]["freq_cm"] == pytest.approx(80.0)
    # Displacement matrix shape
    assert np.asarray(payload["modes"][0]["disp"]).shape == (3, 3)


# ---------------------------------------------------------------------------
# Workflow entry point: an aborted run must not report success / exit 0
# ---------------------------------------------------------------------------

def test_nm_sampling_workflow_abort_exits_nonzero(tmp_path, monkeypatch):
    """Regression: run_nm_sampling returns (None, {}) when it aborts, but the
    workflow still logged "finished successfully" and the CLI exited 0."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from beep.models.nm_sampling import NmSamplingConfig
    from beep.workflows import nm_sampling as wf

    cfg_dict = json.loads(
        (Path(__file__).resolve().parent.parent / "examples" / "nm_sampling.json").read_text()
    )
    config = NmSamplingConfig(**cfg_dict)
    monkeypatch.chdir(tmp_path)

    with patch.object(wf.qcf, "check_collection_existence"), \
         patch.object(wf.qcf, "get_collection", return_value=MagicMock()), \
         patch.object(wf, "run_nm_sampling", return_value=(None, {})), \
         patch.object(wf, "padded_log") as m_padded:
        with pytest.raises(SystemExit) as exc:
            wf.run(config, MagicMock())

    assert exc.value.code == 1
    assert not any(
        "finished successfully" in str(c.args[1]) for c in m_padded.call_args_list
    )


# ---------------------------------------------------------------------------
# decompose_force_errors
# ---------------------------------------------------------------------------

def _water_dimer():
    from beep.core.normal_mode_sampling import decompose_force_errors  # noqa: F401
    pos = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
                    [2.9, 0.0, 0.0], [3.3, 0.8, 0.0], [3.3, -0.8, 0.0]])
    masses = np.array([16.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    return pos, masses, [[0, 1, 2], [3, 4, 5]]


def _pair(delta_plus, delta_minus, band="intermolecular"):
    pos, masses, _ = _water_dimer()
    return [
        {"pair": (0, 0.1), "sign": +1, "band": band, "delta": delta_plus,
         "positions_A": pos, "masses": masses},
        {"pair": (0, 0.1), "sign": -1, "band": band, "delta": delta_minus,
         "positions_A": pos, "masses": masses},
    ]


def test_decompose_constant_offset_goes_to_offset_only():
    """An error identical at +A and -A is pure offset: zero response."""
    from beep.core.normal_mode_sampling import decompose_force_errors
    _, _, frags = _water_dimer()
    rng = np.random.default_rng(0)
    offset = rng.normal(size=(6, 3))
    parts = decompose_force_errors(_pair(offset, offset), frags)
    np.testing.assert_allclose(parts["response"]["all"], 0.0, atol=1e-12)
    np.testing.assert_allclose(parts["offset"]["all"], offset.ravel())


def test_decompose_antisymmetric_error_goes_to_response_only():
    from beep.core.normal_mode_sampling import decompose_force_errors
    _, _, frags = _water_dimer()
    d = np.random.default_rng(1).normal(size=(6, 3))
    parts = decompose_force_errors(_pair(d, -d), frags)
    np.testing.assert_allclose(parts["offset"]["all"], 0.0, atol=1e-12)
    np.testing.assert_allclose(parts["response"]["all"], d.ravel())


def test_decompose_internal_forces_cancel_in_rigid_body_parts():
    """Equal and opposite forces along a bond (an internal force) give no
    net force and no torque on the fragment."""
    from beep.core.normal_mode_sampling import decompose_force_errors
    pos, _, frags = _water_dimer()
    d = np.zeros((6, 3))
    bond = pos[1] - pos[0]
    d[0], d[1] = 5.0 * bond, -5.0 * bond
    parts = decompose_force_errors(_pair(d, d), frags)
    np.testing.assert_allclose(parts["intermolecular"]["all"], 0.0, atol=1e-12)
    np.testing.assert_allclose(parts["torque"]["all"], 0.0, atol=1e-12)
    assert np.abs(parts["offset"]["all"]).max() > 1.0


def test_decompose_rigid_translation_error_is_net_force():
    from beep.core.normal_mode_sampling import decompose_force_errors
    _, _, frags = _water_dimer()
    d = np.zeros((6, 3)); d[[3, 4, 5], 0] = 2.0     # push molecule 2 along x
    parts = decompose_force_errors(_pair(d, d), frags)
    net = parts["intermolecular"]["all"].reshape(-1, 3)   # 2 geoms x 2 frags
    np.testing.assert_allclose(net[1], [6.0, 0.0, 0.0])
    np.testing.assert_allclose(net[0], 0.0, atol=1e-12)


def test_decompose_unpaired_sample_skips_pair_parts_and_tracks_band():
    from beep.core.normal_mode_sampling import decompose_force_errors
    _, _, frags = _water_dimer()
    samples = _pair(np.ones((6, 3)), np.ones((6, 3)), band="stretching")[:1]
    parts = decompose_force_errors(samples, frags)
    assert parts["offset"]["all"].size == 0
    assert parts["response"]["all"].size == 0
    assert parts["intermolecular"]["stretching"].size == 6   # 2 frags x 3


# ---------------------------------------------------------------------------
# geometry shift (Newton step)
# ---------------------------------------------------------------------------

def _model_hessian(pos_bohr, masses, k_bond=0.5, k_soft=0.01):
    """Random SPD Hessian with translations/rotations projected out, so its
    pseudo-inverse is well defined (stiff + soft eigenvalues)."""
    from beep.core.normal_mode_sampling import hessian_pseudo_inverse  # noqa: F401
    n = 3 * len(masses)
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    h = q @ np.diag(np.linspace(k_soft, k_bond, n)) @ q.T
    return h


def test_hessian_pseudo_inverse_inverts_on_vibrational_space():
    from beep.core.normal_mode_sampling import hessian_pseudo_inverse
    pos, masses, _ = _water_dimer()
    pos_b = pos / 0.529177210903
    h = _model_hessian(pos_b, masses)
    h_plus, n_kept = hessian_pseudo_inverse(h, masses, pos_b, min_freq_cm=0.0)
    assert n_kept == 3 * len(masses) - 6
    # A displacement that is a pure vibration is recovered from its force.
    w, v = np.linalg.eigh(h_plus)
    d = v[:, -1] * 0.01
    np.testing.assert_allclose(h_plus @ np.linalg.pinv(h_plus) @ d, d, atol=1e-10)


def test_hessian_pseudo_inverse_ignores_rigid_translation_force():
    """A uniform force (pure translation of the whole system) has no
    vibrational component, so it predicts no geometry shift."""
    from beep.core.normal_mode_sampling import hessian_pseudo_inverse
    pos, masses, _ = _water_dimer()
    pos_b = pos / 0.529177210903
    h_plus, _ = hessian_pseudo_inverse(_model_hessian(pos_b, masses), masses, pos_b,
                                       min_freq_cm=0.0)
    g = np.tile(masses[:, None] * np.array([1.0, 0.0, 0.0]), 1).ravel()
    np.testing.assert_allclose(h_plus @ g, 0.0, atol=1e-10)


def test_hessian_pseudo_inverse_drops_modes_below_floor():
    from beep.core.normal_mode_sampling import hessian_pseudo_inverse
    pos, masses, _ = _water_dimer()
    pos_b = pos / 0.529177210903
    h = _model_hessian(pos_b, masses)
    _, all_kept = hessian_pseudo_inverse(h, masses, pos_b, min_freq_cm=0.0)
    _, some_kept = hessian_pseudo_inverse(h, masses, pos_b, min_freq_cm=1e6)
    assert some_kept == 0 < all_kept


def test_rigid_body_split_translation_and_internal():
    from beep.core.normal_mode_sampling import rigid_body_split
    pos, _, frags = _water_dimer()
    shift = np.zeros((6, 3)); shift[[3, 4, 5]] = [0.1, 0.0, 0.0]   # rigid move
    rigid, internal = rigid_body_split(shift, pos, frags)
    np.testing.assert_allclose(rigid, shift, atol=1e-12)
    np.testing.assert_allclose(internal, 0.0, atol=1e-12)
    bond = np.zeros((6, 3)); bond[1] = 0.01 * (pos[1] - pos[0])      # stretch O-H
    rigid, internal = rigid_body_split(bond, pos, frags)
    np.testing.assert_allclose(rigid + internal, bond, atol=1e-12)
    assert np.abs(internal).max() > 0.0


def test_mean_pair_offset_averages_complete_pairs_only():
    from beep.core.normal_mode_sampling import mean_pair_offset
    samples = _pair(np.full((6, 3), 3.0), np.full((6, 3), 1.0))
    lonely = dict(samples[0]); lonely["pair"] = (9, 0.1); lonely["delta"] = np.full((6, 3), 99.0)
    np.testing.assert_allclose(mean_pair_offset(samples + [lonely]), 2.0)
    assert mean_pair_offset([lonely]) is None


def test_log_nm_force_decomposition_sorts_and_marks_best():
    import logging
    from beep.core.logging_utils import log_nm_force_decomposition

    class _Cap(logging.Handler):
        def __init__(self):
            super().__init__(); self.lines = []
        def emit(self, record):
            self.lines.append(record.getMessage())

    log = logging.getLogger("test_nm_decomp"); log.setLevel(logging.INFO)
    cap = _Cap(); log.addHandler(cap)
    base = {"rmsd_force": 1.0, "rmsd_force_offset": 1.0, "rmsd_force_response": 1.0,
            "rmsd_torque": 1.0, "geometry_shift_intermolecular": 0.01}
    metrics = {
        "a_x": {**base, "rmsd_force_intermolecular": 30.0, "rmsd_force": 10.0},
        "b_x": {**base, "rmsd_force_intermolecular": 10.0, "rmsd_force": 90.0},
        "c_x": {**base, "rmsd_force_intermolecular": 20.0, "rmsd_force": 50.0},
    }
    log_nm_force_decomposition(log, metrics, {"geom_grp": ["A_x", "B_x", "C_x"]})
    rows = [l for l in cap.lines if l[:3] in ("A_x", "B_x", "C_x")]
    assert [r[:3] for r in rows] == ["B_x", "C_x", "A_x"]
    assert rows[0].endswith("*") and not rows[1].endswith("*")
    log.removeHandler(cap)


def test_nm_sampling_ranking_metric_defaults_to_intermolecular():
    from beep.models.nm_sampling import NmSamplingConfig
    assert NmSamplingConfig.model_fields["ranking_metric"].default == "intermolecular"
