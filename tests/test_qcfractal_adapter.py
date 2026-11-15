"""Tests for beep/adapters/qcfractal_adapter.py — all server calls mocked."""
import logging
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from qcportal.record_models import RecordStatusEnum

from beep.adapters.qcfractal_adapter import (
    connect,
    get_or_create_opt_dataset,
    check_collection_exists,
    fetch_opt_molecules,
    check_for_completion,
    check_jobs_status,
    create_keyword_set,
    get_job_ids,
    get_zpve_mol,
    query_keywords,
    is_complete,
    is_incomplete,
    is_error,
    status_label,
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@patch("beep.adapters.qcfractal_adapter.PortalClient")
def test_connect_calls_portal_client(mock_portal):
    mock_portal.return_value = MagicMock()
    client = connect("localhost:7777")
    mock_portal.assert_called_once_with(
        address="localhost:7777", verify=False,
        username=None, password=None,
        show_motd=False,
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def test_get_or_create_existing():
    mock_client = MagicMock()
    mock_ds = MagicMock()
    mock_client.get_dataset.return_value = mock_ds
    result = get_or_create_opt_dataset(mock_client, "test_ds")
    assert result is mock_ds


def test_get_or_create_new():
    mock_client = MagicMock()
    mock_ds = MagicMock()
    mock_client.get_dataset.side_effect = KeyError("not found")
    mock_client.add_dataset.return_value = mock_ds

    result = get_or_create_opt_dataset(mock_client, "new_ds")
    mock_client.add_dataset.assert_called_once_with("optimization", "new_ds")
    assert result is mock_ds


def test_check_collection_exists_true():
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = MagicMock()
    assert check_collection_exists(mock_client, "OptimizationDataset", "ds") is True


def test_check_collection_exists_false():
    mock_client = MagicMock()
    mock_client.get_dataset.side_effect = KeyError("not found")
    assert check_collection_exists(mock_client, "OptimizationDataset", "ds") is False


# ---------------------------------------------------------------------------
# Molecule queries
# ---------------------------------------------------------------------------

def test_fetch_opt_molecules_filters_by_status():
    mock_ds = MagicMock()
    sentinel_mol = MagicMock()

    # Two records: one complete, one error
    rec_complete = MagicMock()
    rec_complete.status = RecordStatusEnum.complete
    rec_complete.final_molecule = sentinel_mol

    rec_error = MagicMock()
    rec_error.status = RecordStatusEnum.error

    mock_ds.get_record.side_effect = [rec_complete, rec_error]

    result = fetch_opt_molecules(mock_ds, ["entry1", "entry2"], "opt_lot")
    assert len(result) == 1
    assert result[0][0] == "entry1"
    assert result[0][1] is sentinel_mol


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def test_get_job_ids_skips_missing_entries():
    """get_job_ids must not raise if entry_list contains names that aren't
    in the dataset — pose generation can short-return, leaving requested
    names un-inserted (see reports/bugs/beep_sampling_get_job_ids_missing_entries.md).
    """
    mock_ds = MagicMock()
    mock_ds.entry_names = ["e1", "e2"]  # "e3" is asked for but absent

    rec1 = MagicMock(); rec1.id = 101
    rec2 = MagicMock(); rec2.id = 102
    mock_ds.get_record.side_effect = lambda n, lot: {"e1": rec1, "e2": rec2}[n]

    pids = get_job_ids(mock_ds, ["e1", "e3", "e2"], "opt_lot")

    assert pids == [101, 102]
    # get_record never called for the missing name — saved a server round-trip
    assert mock_ds.get_record.call_count == 2
    called_names = [c.args[0] for c in mock_ds.get_record.call_args_list]
    assert "e3" not in called_names


def test_check_for_completion_all_done():
    mock_client = MagicMock()
    rec1 = MagicMock()
    rec1.status = RecordStatusEnum.complete
    rec2 = MagicMock()
    rec2.status = RecordStatusEnum.complete
    mock_client.get_records.return_value = [rec1, rec2]

    done, counts = check_for_completion(mock_client, [1, 2])
    assert done is True
    assert counts["COMPLETE"] == 2


def test_check_for_completion_incomplete():
    mock_client = MagicMock()
    rec1 = MagicMock()
    rec1.status = RecordStatusEnum.complete
    rec2 = MagicMock()
    rec2.status = RecordStatusEnum.running
    mock_client.get_records.return_value = [rec1, rec2]

    done, counts = check_for_completion(mock_client, [1, 2])
    assert done is False
    assert counts["INCOMPLETE"] == 1


def _service_record(rec_id, status, is_service=True, children_errors=None):
    """Build a mock record for check_jobs_status tests.

    ``children_errors`` controls the conservative auto-recovery path:
    an empty list (default) means "no errored children" → eligible for
    reset; a non-empty list means "real child error present" → must NOT
    be reset.
    """
    r = MagicMock()
    r.id = rec_id
    r.status = status
    r.is_service = is_service
    r.children_errors = [] if children_errors is None else children_errors
    return r


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_auto_recovers_errored_services():
    """When a parent ReactionRecord is at ERROR, check_jobs_status should
    reset it once so the server re-iterates the service. If the children
    have since been fixed, the service transitions to COMPLETE on its own.
    """
    mock_client = MagicMock()
    logger = MagicMock()

    err_svc = _service_record(101, RecordStatusEnum.error, is_service=True)
    done_svc = _service_record(101, RecordStatusEnum.complete, is_service=True)

    # First poll: service is ERROR → should trigger reset.
    # Second poll: server has re-iterated and the service is now COMPLETE → exit.
    mock_client.get_records.side_effect = [[err_svc], [done_svc]]

    check_jobs_status(mock_client, [101], logger, wait_interval=1)

    mock_client.reset_records.assert_called_once_with([101])


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_does_not_reset_leaf_records():
    """Singlepoint / leaf records (is_service=False) must NEVER be
    auto-reset — that would trigger needless recomputation."""
    mock_client = MagicMock()
    logger = MagicMock()

    err_leaf = _service_record(202, RecordStatusEnum.error, is_service=False)
    # Loop exits after one poll because INCOMPLETE==0 and no recoverables.
    mock_client.get_records.return_value = [err_leaf]

    check_jobs_status(mock_client, [202], logger, wait_interval=1)

    mock_client.reset_records.assert_not_called()


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_resets_each_service_only_once():
    """A service stuck at ERROR (real child error) must not be reset on
    every polling cycle. After the first reset, subsequent ERRORs are
    accepted as terminal."""
    mock_client = MagicMock()
    logger = MagicMock()

    err_svc_a = _service_record(303, RecordStatusEnum.error, is_service=True)
    err_svc_b = _service_record(303, RecordStatusEnum.error, is_service=True)
    # Three polls in a row, same service in ERROR; only the first triggers reset.
    mock_client.get_records.side_effect = [[err_svc_a], [err_svc_b], [err_svc_b]]

    check_jobs_status(mock_client, [303], logger, wait_interval=1, max_wait=2)

    # After the second poll, services_to_recover is empty and
    # INCOMPLETE==0, so the loop exits there. Only one reset call.
    assert mock_client.reset_records.call_count == 1
    mock_client.reset_records.assert_called_with([303])


def test_get_zpve_mol_filters_incomplete_records():
    """get_zpve_mol must skip hessian records whose `properties` is None —
    a 'complete'-tagged record can still be mid-write or otherwise
    inconsistent, and `result.return_result` would crash on it.

    Regression for the case where multiple hessian records exist for the
    same molecule + spec and the first one is unusable.
    """
    mock_client = MagicMock()

    # Mock molecule with multiple symbols so we skip the atom early-return.
    mock_mol = MagicMock()
    mock_mol.symbols = ["O", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H"]
    mock_mol.dict.return_value = {"identifiers": {"molecular_formula": "H10O"}}
    mock_client.get_molecules.return_value = [mock_mol]

    # Bad record: complete-tagged but properties is None (server inconsistency).
    bad_record = MagicMock()
    bad_record.properties = None

    mock_client.query_singlepoints.return_value = iter([bad_record])

    # With our filter, this should hit the "no usable record" branch and
    # return (None, True) instead of crashing on result.return_result.
    zpve, ok = get_zpve_mol(mock_client, 12345, "mpwb1k-d3bj_def2-tzvpd")
    assert zpve is None
    assert ok is True

    # Verify the query was made with status=complete, narrowing to
    # server-side completes before we even client-side filter.
    kwargs = mock_client.query_singlepoints.call_args_list[0].kwargs
    assert kwargs["status"] == RecordStatusEnum.complete


def _zpve_mock_client():
    mock_client = MagicMock()
    mock_mol = MagicMock()
    mock_mol.symbols = ["O", "H", "H"]
    mock_mol.dict.return_value = {"identifiers": {"molecular_formula": "H2O"}}
    mock_client.get_molecules.return_value = [mock_mol]
    return mock_client


def test_get_zpve_mol_finds_orca_bare_method_hessian():
    """ORCA Hessians of dispersion LOTs are stored under the bare functional
    with the dispersion keyword in simple_input; get_zpve_mol must fall back
    to that query shape when the compound-method query returns nothing.

    The fallback queries every program in DISPERSION_KEYWORD_SPEC that knows
    the suffix, in insertion order (orca, then gaussian for -d3bj)."""
    mock_client = _zpve_mock_client()

    orca_record = MagicMock()
    orca_record.specification.keywords = {"simple_input": "D3BJ"}
    # calls: compound-method (empty), orca bare-method (hit), gaussian bare-method (empty)
    mock_client.query_singlepoints.side_effect = [iter([]), iter([orca_record]), iter([])]

    # Interrupt before the vibrational analysis: we only care about record lookup
    with patch("beep.core.zpve._vibanal_wfn", side_effect=RuntimeError("stop")):
        with pytest.raises(RuntimeError, match="stop"):
            get_zpve_mol(mock_client, 12345, "mpwb1k-d3bj_def2-tzvpd")

    calls = mock_client.query_singlepoints.call_args_list
    assert calls[0].kwargs["method"] == "mpwb1k-d3bj"
    assert calls[1].kwargs["method"] == "mpwb1k"
    assert calls[1].kwargs["program"] == "orca"
    assert calls[2].kwargs["program"] == "gaussian"


def test_get_zpve_mol_finds_gaussian_bare_method_hessian():
    """Gaussian Hessians store the dispersion keyword in route_input
    (e.g. 'EmpiricalDispersion=GD3BJ'); the token filter must accept the
    matching record and reject a different-damping one (GD3 vs GD3BJ)."""
    mock_client = _zpve_mock_client()

    gau_record = MagicMock()
    gau_record.specification.keywords = {"route_input": "EmpiricalDispersion=GD3BJ SCF=Tight"}
    wrong_damping = MagicMock()
    wrong_damping.specification.keywords = {"route_input": "EmpiricalDispersion=GD3"}
    # calls: compound-method (empty), orca (empty), gaussian (one hit + one filtered out)
    mock_client.query_singlepoints.side_effect = [iter([]), iter([]), iter([wrong_damping, gau_record])]

    with patch("beep.core.zpve._vibanal_wfn", side_effect=RuntimeError("stop")):
        with pytest.raises(RuntimeError, match="stop"):
            get_zpve_mol(mock_client, 12345, "b3lyp-d3bj_def2-svp")

    calls = mock_client.query_singlepoints.call_args_list
    assert calls[2].kwargs["method"] == "b3lyp"
    assert calls[2].kwargs["program"] == "gaussian"


def test_keyword_token_match():
    from beep.adapters.qcfractal_adapter import _keyword_token_match

    assert _keyword_token_match("D3BJ", "D3BJ")
    assert _keyword_token_match("d3bj tightscf", "D3BJ")
    assert _keyword_token_match("EmpiricalDispersion=GD3BJ SCF=Tight", "EmpiricalDispersion=GD3BJ")
    # exact-token: GD3 must not match GD3BJ and vice versa
    assert not _keyword_token_match("EmpiricalDispersion=GD3BJ", "EmpiricalDispersion=GD3")
    assert not _keyword_token_match("EmpiricalDispersion=GD3", "EmpiricalDispersion=GD3BJ")
    assert not _keyword_token_match("", "D4")


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_loop_exits_when_reset_fails():
    """Regression for `reports/BUG_check_jobs_status_infinite_loop.md`.

    If `client.reset_records` raises (e.g. transient HTTP 500), the
    service IDs must still be marked as attempted so the same IDs aren't
    re-selected forever. Otherwise the workflow gets wedged at
    `INCOMPLETE == 0, ERROR == N` and runs until ``max_wait``.
    """
    mock_client = MagicMock()
    logger = MagicMock()

    err_svc_a = _service_record(505, RecordStatusEnum.error, is_service=True)
    err_svc_b = _service_record(505, RecordStatusEnum.error, is_service=True)
    # Two polls: same service ERROR both times. After the first reset
    # attempt fails, the service must NOT be re-attempted on the second.
    mock_client.get_records.side_effect = [[err_svc_a], [err_svc_b]]
    mock_client.reset_records.side_effect = RuntimeError(
        "Request failed: HTTP 500 (simulated server overload)"
    )

    # If the bug were present, this would hit max_wait and raise TimeoutError.
    check_jobs_status(mock_client, [505], logger, wait_interval=1, max_wait=10)

    # reset_records called exactly once — second cycle skipped the ID
    assert mock_client.reset_records.call_count == 1


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_skips_services_with_errored_children():
    """Conservative auto-recovery: a parent service whose children are
    themselves in ERROR has a *real* failure (data missing). Resetting it
    is useless at best and wasteful at worst (server-side cascade
    re-runs the failing child). Leave the parent alone.
    """
    mock_client = MagicMock()
    logger = MagicMock()

    # Parent at ERROR, with one child in error (chemistry-error scenario).
    bad_child = MagicMock()
    err_svc = _service_record(
        707, RecordStatusEnum.error,
        is_service=True, children_errors=[bad_child],
    )
    mock_client.get_records.return_value = [err_svc]

    check_jobs_status(mock_client, [707], logger, wait_interval=1)

    # No reset attempt: children_errors non-empty → conservative skip
    mock_client.reset_records.assert_not_called()


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_picks_up_child_reset_without_workflow_restart():
    """When the user resets an errored child externally (mid-workflow),
    the next polling cycle must notice that children_errors is now
    empty and reset the parent. This matches the qcportal-0.15 UX where
    resetting a record made the workflow continue automatically without
    a be_hess restart.

    Models a realistic poll: one stuck-ERROR service the user is fixing
    (id 808) plus one INCOMPLETE service (id 809) that keeps the loop
    alive across cycles. Without 809, the loop would exit at cycle 1
    before the user's reset takes effect.
    """
    mock_client = MagicMock()
    logger = MagicMock()

    bad_child = MagicMock()
    # 808: stuck ERROR, child still in error → skip (no reset).
    svc_stuck = _service_record(
        808, RecordStatusEnum.error,
        is_service=True, children_errors=[bad_child],
    )
    # 808 again, user has reset the child externally → children_errors empty.
    svc_clean = _service_record(
        808, RecordStatusEnum.error,
        is_service=True, children_errors=[],
    )
    # 808 has re-iterated → COMPLETE.
    svc_done = _service_record(
        808, RecordStatusEnum.complete, is_service=True,
    )
    # 809 keeps the loop alive — INCOMPLETE for the first two cycles,
    # then COMPLETE on the third.
    other_running = _service_record(
        809, RecordStatusEnum.running, is_service=True,
    )
    other_done = _service_record(
        809, RecordStatusEnum.complete, is_service=True,
    )

    mock_client.get_records.side_effect = [
        [svc_stuck, other_running],   # cycle 1: 808 skipped, 809 running → keep polling
        [svc_clean, other_running],   # cycle 2: user reset child → reset 808
        [svc_done, other_done],       # cycle 3: both COMPLETE → exit
    ]

    check_jobs_status(mock_client, [808, 809], logger, wait_interval=1)

    # Reset happened exactly once — when 808's children became clean.
    # 809 was never reset (it was running, not errored).
    mock_client.reset_records.assert_called_once_with([808])


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_auto_recover_disabled():
    """Setting auto_recover_services=False preserves the old behavior:
    ERROR is fully terminal, no reset attempts."""
    mock_client = MagicMock()
    logger = MagicMock()

    err_svc = _service_record(404, RecordStatusEnum.error, is_service=True)
    mock_client.get_records.return_value = [err_svc]

    check_jobs_status(
        mock_client, [404], logger, wait_interval=1,
        auto_recover_services=False,
    )

    mock_client.reset_records.assert_not_called()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def test_status_helpers():
    assert is_complete(RecordStatusEnum.complete) is True
    assert is_complete(RecordStatusEnum.error) is False
    assert is_incomplete(RecordStatusEnum.running) is True
    assert is_incomplete(RecordStatusEnum.waiting) is True
    assert is_incomplete(RecordStatusEnum.complete) is False
    assert is_error(RecordStatusEnum.error) is True
    assert is_error(RecordStatusEnum.complete) is False


def test_status_label_mapping():
    assert status_label(RecordStatusEnum.complete) == "COMPLETE"
    assert status_label(RecordStatusEnum.running) == "INCOMPLETE"
    assert status_label(RecordStatusEnum.waiting) == "INCOMPLETE"
    assert status_label(RecordStatusEnum.error) == "ERROR"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

def test_reexports_are_importable():
    """Adapter must re-export QCPortal types so workflows never import
    qcportal directly."""
    from beep.adapters.qcfractal_adapter import (
        FractalClient,
        PortalClient,
        Dataset,
        OptimizationDataset,
        ReactionDataset,
        Molecule,
    )
    assert FractalClient is PortalClient
    for cls in (FractalClient, Dataset, OptimizationDataset, ReactionDataset, Molecule):
        assert isinstance(cls, type), f"{cls} is not a class"


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def test_create_keyword_set_returns_dict():
    result = create_keyword_set({"reference": "uks"})
    assert result == {"reference": "uks"}
    assert isinstance(result, dict)


def test_query_keywords_raises():
    mock_client = MagicMock()
    with pytest.raises(NotImplementedError):
        query_keywords(mock_client)


# ---------------------------------------------------------------------------
# Atom handling
# ---------------------------------------------------------------------------

def test_fetch_atom_molecule():
    from beep.adapters.qcfractal_adapter import fetch_atom_molecule

    mock_client = MagicMock()
    mock_ds = MagicMock()
    sentinel_mol = MagicMock()
    mock_entry = MagicMock()
    mock_entry.molecule = sentinel_mol

    mock_client.get_dataset.return_value = mock_ds
    mock_ds.get_entry.return_value = mock_entry

    result = fetch_atom_molecule(mock_client, "atoms", "O")
    mock_client.get_dataset.assert_called_once_with("singlepoint", "atoms")
    mock_ds.get_entry.assert_called_once_with("O")
    assert result is sentinel_mol


def test_fetch_atom_molecule_not_found():
    from beep.adapters.qcfractal_adapter import fetch_atom_molecule

    mock_client = MagicMock()
    mock_ds = MagicMock()
    mock_client.get_dataset.return_value = mock_ds
    mock_ds.get_entry.return_value = None

    with pytest.raises(KeyError, match="not found"):
        fetch_atom_molecule(mock_client, "atoms", "Xe")


# ---------------------------------------------------------------------------
# Dispersion splitting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, expected",
    [
        ("pbe-d3bj",    ("pbe",    "pbe-d3bj",    "dftd3")),
        ("b3lyp-d3mbj", ("b3lyp",  "b3lyp-d3mbj", "dftd3")),
        ("mpwb1k-d3m",  ("mpwb1k", "mpwb1k-d3m",  "dftd3")),
        ("pbe-d3",      ("pbe",    "pbe-d3",      "dftd3")),
        ("b3lyp-d4",    ("b3lyp",  "b3lyp-d4",    "dftd4")),
        ("PBE-D3BJ",    ("PBE",    "PBE-D3BJ",    "dftd3")),  # case preserved in bare/full
        ("wb97x-v",     ("wb97x-v", None, None)),  # intrinsic dispersion, not split
        ("wb97m-v",     ("wb97m-v", None, None)),
        ("hf3c",        ("hf3c",   None, None)),
        ("pbe",         ("pbe",    None, None)),
    ],
)
def test_split_dispersion(method, expected):
    from beep.adapters.qcfractal_adapter import _split_dispersion
    assert _split_dispersion(method) == expected


@pytest.mark.parametrize(
    "program, expected",
    [
        ("dftd3", "s-dftd3"),   # legacy executable harness has no pbc support
        ("dftd4", "dftd4"),     # already the python-API harness
        ("s-dftd3", "s-dftd3"), # idempotent
        (None, None),
    ],
)
def test_periodic_dispersion_program(program, expected):
    """Periodic runs must not use QCEngine's legacy 'dftd3' harness: it ignores
    keywords['cell']/['pbc'] and would silently return cluster dispersion for a slab."""
    from beep.adapters.qcfractal_adapter import periodic_dispersion_program
    assert periodic_dispersion_program(program) == expected


def test_periodic_dispersion_leaves_cluster_routing_untouched():
    """The cluster path keeps 'dftd3' so existing dispersion specs/records stay valid."""
    from beep.adapters.qcfractal_adapter import _split_dispersion
    assert _split_dispersion("mpwb1k-d3bj")[2] == "dftd3"


@pytest.mark.parametrize(
    "method, mult, program, expected",
    [
        # psi4: method passes through (psi4 parses dispersion suffixes itself)
        ("b3lyp-d4", 1, "psi4", ("b3lyp-d4", {"function_kwargs": {"dertype": 1}})),
        ("mpwb1k-d3bj", 2, "psi4",
         ("mpwb1k-d3bj", {"function_kwargs": {"dertype": 1}, "reference": "uks"})),
        ("pbe", 1, "psi4", ("pbe", {"function_kwargs": {"dertype": 1}})),
        # orca: bare functional + native dispersion keyword via simple_input;
        # no psi4 keywords, UKS implicit
        ("b3lyp-d4", 1, "orca", ("b3lyp", {"simple_input": "D4"})),
        ("mpwb1k-d3bj", 2, "orca", ("mpwb1k", {"simple_input": "D3BJ"})),
        ("pbe-d3", 1, "orca", ("pbe", {"simple_input": "D3ZERO"})),
        ("pbe", 1, "orca", ("pbe", {})),
        ("b3lyp-d4", 1, "ORCA", ("b3lyp", {"simple_input": "D4"})),  # case-insensitive program
        # gaussian: bare functional + native dispersion keyword via route_input;
        # no psi4 keywords, UHF/UKS implicit
        ("b3lyp-d3bj", 1, "gaussian", ("b3lyp", {"route_input": "EmpiricalDispersion=GD3BJ"})),
        ("pbe-d3", 2, "gaussian", ("pbe", {"route_input": "EmpiricalDispersion=GD3"})),
        ("pbe", 1, "gaussian", ("pbe", {})),
        ("b3lyp-d3bj", 1, "Gaussian", ("b3lyp", {"route_input": "EmpiricalDispersion=GD3BJ"})),
        # mace: opt-LOT alias -> model file, no keywords (harness default: full fd Hessian)
        ("lmft-ch4-b3lypd4-v4", 1, "mace", ("lmft-ch4-b3lypd4-v4.model", {})),
        ("mace-off23-large.model", 2, "MACE", ("mace-off23-large.model", {})),
    ],
)
def test_hessian_method_and_keywords(method, mult, program, expected):
    from beep.adapters.qcfractal_adapter import hessian_method_and_keywords
    assert hessian_method_and_keywords(method, mult, program) == expected


@pytest.mark.parametrize(
    "method, program",
    [
        ("b3lyp-d3m", "orca"),
        ("b3lyp-d3mbj", "orca"),
        ("b3lyp-d4", "gaussian"),  # G16 has no D4
        ("b3lyp-d3m", "gaussian"),
        ("b3lyp-d3mbj", "gaussian"),
    ],
)
def test_hessian_method_and_keywords_unsupported_damping(method, program):
    from beep.adapters.qcfractal_adapter import hessian_method_and_keywords
    with pytest.raises(ValueError, match="no native"):
        hessian_method_and_keywords(method, 1, program)


# ---------------------------------------------------------------------------
# compute_be_dft_energies — dispatch to separated-pair vs integrated form
# ---------------------------------------------------------------------------

def _fake_submit_result(n_inserted=1, n_existing=0):
    r = MagicMock()
    r.n_inserted = n_inserted
    r.n_existing = n_existing
    return r


@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_splits_dispersion(mock_submit):
    """D3BJ method → two submit_energies calls per stoich: bare DFT + bare dispersion."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies, STOICH_TYPES

    mock_submit.return_value = _fake_submit_result()
    logger = MagicMock()

    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_H2O_W5_01",
        all_dft=["pbe-d3bj_def2-tzvp"], tag="test",
        program="psi4", logger=logger,
    )

    # Two calls per stoich type: bare DFT (psi4) + bare dispersion (dftd3)
    assert mock_submit.call_count == 2 * len(STOICH_TYPES)

    dft_calls = [c for c in mock_submit.call_args_list if c.kwargs["program"] == "psi4"]
    disp_calls = [c for c in mock_submit.call_args_list if c.kwargs["program"] == "dftd3"]
    assert len(dft_calls) == len(STOICH_TYPES)
    assert len(disp_calls) == len(STOICH_TYPES)

    assert dft_calls[0].kwargs["method"] == "pbe"
    assert dft_calls[0].kwargs["basis"] == "def2-tzvp"
    assert disp_calls[0].kwargs["method"] == "pbe-d3bj"
    assert disp_calls[0].kwargs["basis"] is None


@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_integrated_for_non_dispersion(mock_submit):
    """HF-3c has no dispersion suffix → single integrated spec per stoich.
    HF-3c is a -3c composite (gCP built into the SCF), so the ``bsse``
    stoich is intentionally skipped — counterpoise on top would
    double-correct."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies, STOICH_TYPES

    mock_submit.return_value = _fake_submit_result()
    logger = MagicMock()

    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_H2O_W5_01",
        all_dft=["hf3c_minix"], tag="test",
        program="psi4", logger=logger,
    )

    # All stoichs except ``bsse`` get submitted for -3c methods.
    expected_stoichs = [s for s in STOICH_TYPES if s != "bsse"]
    assert mock_submit.call_count == len(expected_stoichs)
    called_stoichs = [call.kwargs["stoich"] for call in mock_submit.call_args_list]
    assert "bsse" not in called_stoichs
    assert sorted(called_stoichs) == sorted(expected_stoichs)
    for call in mock_submit.call_args_list:
        assert call.kwargs["method"] == "hf3c"
        assert call.kwargs["basis"] == "minix"
        assert call.kwargs["program"] == "psi4"


@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_d4_uses_dftd4_program(mock_submit):
    """D4 dispersion routes to dftd4 program (not dftd3)."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies

    mock_submit.return_value = _fake_submit_result()

    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_H2O_W5_01",
        all_dft=["b3lyp-d4_def2-tzvp"], tag="test",
        program="psi4", logger=MagicMock(),
    )

    disp_calls = [c for c in mock_submit.call_args_list if c.kwargs["program"] == "dftd4"]
    assert len(disp_calls) > 0
    assert disp_calls[0].kwargs["method"] == "b3lyp-d4"
    assert disp_calls[0].kwargs["basis"] is None


@pytest.mark.parametrize("lot", ["wb97m-d3bj_def2-tzvppd", "B97M-D3BJ_def2-tzvp",
                                 "b97-d3bj_def2-tzvp", "b97-d3mbj_def2-tzvp"])
@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_integrated_dispersion_methods(mock_submit, lot):
    """Functionals without a bare psi4 form → one integrated psi4 spec per stoich,
    no dispersion-only spec."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies, STOICH_TYPES

    mock_submit.return_value = _fake_submit_result()
    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_CH4_W12_10",
        all_dft=[lot], tag="test", program="psi4", logger=MagicMock(),
    )
    method, basis = lot.split("_", 1)
    assert mock_submit.call_count == len(STOICH_TYPES)
    for call in mock_submit.call_args_list:
        assert call.kwargs["program"] == "psi4"
        assert call.kwargs["method"] == method
        assert call.kwargs["basis"] == basis


def test_integrated_dispersion_methods_match_psi4():
    """The list equals psi4's suffixed functionals whose bare name psi4 lacks."""
    functionals = pytest.importorskip(
        "psi4.driver.procrouting.dft").functionals
    from beep.adapters.qcfractal_adapter import (
        DISPERSION_SUFFIXES, INTEGRATED_DISPERSION_METHODS)

    expected = set()
    for name in functionals:
        for suffix in DISPERSION_SUFFIXES:
            if name.endswith(suffix):
                if name[: -len(suffix)] not in functionals:
                    expected.add(name)
                break
    assert expected == set(INTEGRATED_DISPERSION_METHODS)


# ---------------------------------------------------------------------------
# fetch_reaction_values — separated-pair recombination, integrated-spec skip
# ---------------------------------------------------------------------------


@patch("beep.adapters.qcfractal_adapter._stoich_dataset_name",
       return_value="be_CH4_W12_10_bsse")
def test_fetch_reaction_values_reads_integrated_dispersion_method(mock_ds_name):
    """wb97m-d3bj is stored integrated and comes back as its own column, next
    to a separated pair that is still summed as before."""
    from beep.adapters.qcfractal_adapter import fetch_reaction_values
    import qcelemental as qcel

    records = {
        "wb97m-d3bj_def2-tzvppd": {"entry1": -2.0},
        "b3lyp_def2-tzvpd":       {"entry1": -1.0},
        "b3lyp-d4":               {"entry1": -0.01},
    }
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = _fake_reaction_dataset(records)
    df = fetch_reaction_values(mock_client, "be_CH4_W12_10", stoich="bsse")

    conv = qcel.constants.hartree2kcalmol
    assert sorted(df.columns) == ["b3lyp-d4/def2-tzvpd", "wb97m-d3bj/def2-tzvppd"]
    assert df.loc["entry1", "wb97m-d3bj/def2-tzvppd"] == pytest.approx(-2.0 * conv)
    assert df.loc["entry1", "b3lyp-d4/def2-tzvpd"] == pytest.approx(-1.01 * conv)


def _fake_reaction_dataset(records_by_spec):
    """Build a mock ReactionDataset for fetch_reaction_values tests.

    ``records_by_spec`` is a dict: spec_name → {entry_name: total_energy_hartree}.
    """
    ds = MagicMock()
    ds.specification_names = list(records_by_spec.keys())

    def _iterate(specification_names, status):
        sname = specification_names[0]
        for entry_name, energy in records_by_spec[sname].items():
            rec = MagicMock()
            rec.total_energy = energy
            yield entry_name, sname, rec

    ds.iterate_records.side_effect = _iterate
    return ds


@patch("beep.adapters.qcfractal_adapter._stoich_dataset_name",
       return_value="be_H2O_W5_01_bsse")
def test_fetch_reaction_values_d4_separated_pair(mock_ds_name):
    """Separated pair for D4 recombines into composite column."""
    from beep.adapters.qcfractal_adapter import fetch_reaction_values

    # In hartree; composite column should be sum of the two pieces
    records = {
        "b3lyp_def2-tzvp": {"entry1": -1.0},
        "b3lyp-d4":         {"entry1": -0.01},
    }
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = _fake_reaction_dataset(records)

    df = fetch_reaction_values(mock_client, "be_H2O_W5_01", stoich="bsse")

    assert "b3lyp-d4/def2-tzvp" in df.columns
    # Sum in kcal/mol (hartree2kcalmol ≈ 627.5)
    import qcelemental as qcel
    expected = (-1.0 - 0.01) * qcel.constants.hartree2kcalmol
    assert df.loc["entry1", "b3lyp-d4/def2-tzvp"] == pytest.approx(expected)


@patch("beep.adapters.qcfractal_adapter._stoich_dataset_name",
       return_value="be_H2O_W5_01_bsse")
def test_fetch_reaction_values_skips_integrated_spec(mock_ds_name, caplog):
    """Integrated specs (dispersion suffix before underscore) are filtered out
    in favour of the canonical separated bare-DFT + bare-dispersion pair.
    A single summary warning reports the skip count."""
    from beep.adapters.qcfractal_adapter import fetch_reaction_values

    records = {
        "pbe_def2-tzvp":           {"entry1": -1.0},
        "pbe-d3bj":                {"entry1": -0.02},
        "pbe-d3bj_def2-tzvp":      {"entry1": -99.0},   # integrated — skipped
    }
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = _fake_reaction_dataset(records)

    import logging
    beep_logger = logging.getLogger("beep")
    prev_propagate = beep_logger.propagate
    beep_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="beep"):
            df = fetch_reaction_values(mock_client, "be_H2O_W5_01", stoich="bsse")
    finally:
        beep_logger.propagate = prev_propagate

    # Composite column came from the separated pair, not the integrated value
    import qcelemental as qcel
    conv = qcel.constants.hartree2kcalmol
    assert "pbe-d3bj/def2-tzvp" in df.columns
    assert df.loc["entry1", "pbe-d3bj/def2-tzvp"] == pytest.approx(
        (-1.0 - 0.02) * conv
    )
    # Summary warning reports the skip
    assert any(
        "Skipped 1 integrated-dispersion spec" in r.message
        for r in caplog.records
    )

    # Composite column comes from the separated pair, not the skipped integrated value
    import qcelemental as qcel
    expected = (-1.0 - 0.02) * qcel.constants.hartree2kcalmol
    assert df.loc["entry1", "pbe-d3bj/def2-tzvp"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# fetch_opt_record / fetch_atom_molecule — PortalRequestError translation
# ---------------------------------------------------------------------------

def test_fetch_opt_record_translates_missing_entry_to_keyerror():
    """qcportal raises PortalRequestError('Missing N entries: ...') when an
    entry name isn't in the dataset. The adapter must translate that one
    specific case to KeyError so workflow code can route to the
    atoms_collection fallback."""
    from beep.adapters.qcfractal_adapter import fetch_opt_record, PortalRequestError

    ds = MagicMock()
    ds.get_record.side_effect = PortalRequestError(
        "Request failed: Missing 1 entries: H_atom (HTTP status 400)",
        status_code=400,
        details={},
    )
    with pytest.raises(KeyError, match="not found in dataset"):
        fetch_opt_record(ds, "H_atom", "hf3c_minix")


def test_fetch_opt_record_propagates_other_portal_errors():
    """Non-'Missing entries' PortalRequestErrors (server 500, network, auth)
    must propagate unchanged so callers don't mistake transient failures
    for missing entries."""
    from beep.adapters.qcfractal_adapter import fetch_opt_record, PortalRequestError

    ds = MagicMock()
    ds.get_record.side_effect = PortalRequestError(
        "Internal Server Error (HTTP status 500)",
        status_code=500,
        details={},
    )
    with pytest.raises(PortalRequestError):
        fetch_opt_record(ds, "H2O", "hf3c_minix")


def test_fetch_opt_record_raises_keyerror_when_record_none():
    """Existing behaviour preserved: when ds.get_record returns None
    (entry exists but spec record missing), still raise KeyError."""
    from beep.adapters.qcfractal_adapter import fetch_opt_record

    ds = MagicMock()
    ds.get_record.return_value = None
    with pytest.raises(KeyError, match="No record for entry"):
        fetch_opt_record(ds, "H2O", "hf3c_minix")


def test_fetch_atom_molecule_translates_missing_entry_to_keyerror():
    """Mirror of fetch_opt_record for fetch_atom_molecule: PortalRequestError
    on missing atom name must become KeyError so the workflow's outer
    KeyError handler can surface the original 'not optimized' message."""
    from beep.adapters.qcfractal_adapter import fetch_atom_molecule, PortalRequestError

    client = MagicMock()
    ds = MagicMock()
    ds.get_entry.side_effect = PortalRequestError(
        "Request failed: Missing 1 entries: Xe (HTTP status 400)",
        status_code=400,
        details={},
    )
    client.get_dataset.return_value = ds
    with pytest.raises(KeyError, match="not found in singlepoint dataset"):
        fetch_atom_molecule(client, "atoms", "Xe")


def test_fetch_atom_molecule_propagates_other_portal_errors():
    from beep.adapters.qcfractal_adapter import fetch_atom_molecule, PortalRequestError

    client = MagicMock()
    ds = MagicMock()
    ds.get_entry.side_effect = PortalRequestError(
        "Unauthorized (HTTP status 401)",
        status_code=401,
        details={},
    )
    client.get_dataset.return_value = ds
    with pytest.raises(PortalRequestError):
        fetch_atom_molecule(client, "atoms", "H")


# ---------------------------------------------------------------------------
# compute_hessian — basis-less LOTs and program/dispersion-aware reuse
# ---------------------------------------------------------------------------

def _hessian_client(query_results=None):
    """Client whose be_nocp dataset holds one entry over molecules 1 and 2."""
    from beep.adapters.qcfractal_adapter import PortalRequestError  # noqa: F401

    client = MagicMock()
    ds_nocp = MagicMock()
    stoich_1 = MagicMock(); stoich_1.molecule.id = 1
    stoich_2 = MagicMock(); stoich_2.molecule.id = 2
    entry = MagicMock(); entry.stoichiometries = [stoich_1, stoich_2]
    ds_nocp.iterate_entries.return_value = [entry]
    client.get_dataset.return_value = ds_nocp

    mol1 = MagicMock(); mol1.id = 1; mol1.symbols = ["O", "H", "H"]
    mol2 = MagicMock(); mol2.id = 2; mol2.symbols = ["C", "O"]
    client.get_molecules.return_value = [mol1, mol2]

    client.query_singlepoints.return_value = iter(query_results or [])
    meta = MagicMock(); meta.n_existing = 0; meta.n_inserted = 1
    client.add_singlepoints.return_value = (meta, [900])
    return client


@pytest.mark.parametrize("opt_lot, exp_method, exp_basis", [
    ("gfn2-xtb", "gfn2-xtb", None),
    ("b3lyp-d3bj_def2-svp", "b3lyp-d3bj", "def2-svp"),
])
def test_compute_hessian_accepts_basisless_lot(opt_lot, exp_method, exp_basis):
    """``method, basis = opt_lot.split("_")`` used to raise ValueError for
    basis-less LOTs (gfn2-xtb, MACE aliases); only the first underscore
    separates method from basis."""
    from beep.adapters.qcfractal_adapter import compute_hessian

    client = _hessian_client()
    ids = compute_hessian(client, "be_CO_W5_01", opt_lot, 1, "hess", MagicMock())

    assert ids == [900]
    q = client.query_singlepoints.call_args.kwargs
    assert q["method"] == exp_method and q["basis"] == exp_basis
    sub = client.add_singlepoints.call_args.kwargs
    assert sub["method"] == exp_method and sub["basis"] == exp_basis
    assert sorted(sub["molecules"]) == [1, 2]


def _hess_record(rec_id, mol_id, keywords):
    r = MagicMock()
    r.id = rec_id
    r.molecule_id = mol_id
    r.properties = {"return_energy": -1.0}
    r.specification.keywords = keywords
    return r


def test_compute_hessian_reuse_requires_matching_program_and_dispersion():
    """The reuse pre-query must be scoped to ``program`` and, for orca /
    gaussian, to the native dispersion token get_zpve_mol later requires.
    A stray psi4 Hessian (no simple_input token) at the same method/basis
    must not mark an ORCA molecule as done."""
    from beep.adapters.qcfractal_adapter import compute_hessian

    psi4_like = _hess_record(10, 1, {"function_kwargs": {"dertype": 1}})
    orca_like = _hess_record(11, 2, {"simple_input": "D3BJ"})
    client = _hessian_client([psi4_like, orca_like])

    ids = compute_hessian(
        client, "be_CO_W5_01", "b3lyp-d3bj_def2-svp", 1, "hess", MagicMock(),
        program="orca",
    )

    q = client.query_singlepoints.call_args.kwargs
    assert q["program"] == "orca"
    assert q["method"] == "b3lyp"            # bare functional on orca
    # Only molecule 2 (matching D3BJ token) is reused; molecule 1 resubmitted.
    sub = client.add_singlepoints.call_args.kwargs
    assert sub["molecules"] == [1]
    assert sub["program"] == "orca"
    assert sub["keywords"] == {"simple_input": "D3BJ"}
    assert ids == [11, 900]


def test_compute_hessian_reuse_psi4_rejects_dispersion_keyword_records():
    """On the psi4 path a record whose keywords carry a native dispersion
    token belongs to another program's spec and must not count as done."""
    from beep.adapters.qcfractal_adapter import compute_hessian

    orca_like = _hess_record(11, 1, {"simple_input": "D3BJ"})
    plain = _hess_record(12, 2, {"function_kwargs": {"dertype": 1}})
    client = _hessian_client([orca_like, plain])

    ids = compute_hessian(
        client, "be_CO_W5_01", "b3lyp-d3bj_def2-svp", 1, "hess", MagicMock(),
        program="psi4",
    )
    assert client.query_singlepoints.call_args.kwargs["program"] == "psi4"
    assert client.add_singlepoints.call_args.kwargs["molecules"] == [1]
    assert ids == [12, 900]


def test_compute_hessian_orca_bare_functional_rejects_dispersion_records():
    """A plain (non-dispersion) orca request must not reuse a b3lyp-d3bj
    orca Hessian, which is stored under the same bare method."""
    from beep.adapters.qcfractal_adapter import compute_hessian

    disp = _hess_record(11, 1, {"simple_input": "D3BJ"})
    bare = _hess_record(12, 2, {})
    client = _hessian_client([disp, bare])

    compute_hessian(
        client, "be_CO_W5_01", "b3lyp_def2-svp", 1, "hess", MagicMock(),
        program="orca",
    )
    assert client.add_singlepoints.call_args.kwargs["molecules"] == [1]


# ---------------------------------------------------------------------------
# _collect_reaction_record_ids — restricted to the specs just submitted
# ---------------------------------------------------------------------------

def test_collect_reaction_record_ids_filters_by_spec_names():
    from beep.adapters.qcfractal_adapter import _collect_reaction_record_ids

    client = MagicMock()
    ds = MagicMock()
    rec = MagicMock(); rec.id = 7
    ds.iterate_records.return_value = [("e1", "pbe_def2-svp", rec)]
    client.get_dataset.return_value = ds

    ids = _collect_reaction_record_ids(
        client, "be_H2O_W5_01", ("be_nocp",), spec_names=["PBE_def2-svp", "pbe-d3bj"],
    )
    assert ids == [7]
    kwargs = ds.iterate_records.call_args.kwargs
    assert kwargs["specification_names"] == ["pbe-d3bj", "pbe_def2-svp"]


@patch("beep.adapters.qcfractal_adapter._collect_reaction_record_ids")
@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_monitors_only_submitted_specs(mock_submit, mock_collect):
    """check_jobs_status must not wait on LOTs of earlier runs: the record
    collection is restricted to the spec names built in this call."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies

    mock_submit.return_value = _fake_submit_result()
    mock_collect.return_value = []
    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_H2O_W5_01",
        all_dft=["pbe-d3bj_def2-tzvp", "hf3c_minix"], tag="t",
        program="psi4", logger=MagicMock(),
    )
    spec_names = mock_collect.call_args.kwargs["spec_names"]
    assert sorted(set(spec_names)) == ["hf3c_minix", "pbe-d3bj", "pbe_def2-tzvp"]


# ---------------------------------------------------------------------------
# create_or_load_reaction_dataset — only COMPLETE records, count on success
# ---------------------------------------------------------------------------

@patch("beep.adapters.qcfractal_adapter.add_reaction")
@patch("beep.adapters.qcfractal_adapter.be_stoichiometry")
def test_create_or_load_reaction_dataset_skips_non_complete(mock_stoich, mock_add):
    from beep.adapters.qcfractal_adapter import create_or_load_reaction_dataset

    def rec(status):
        r = MagicMock(); r.status = status
        r.final_molecule = MagicMock() if status == RecordStatusEnum.complete else None
        return r

    ds_opt = MagicMock()
    ds_opt.get_record.side_effect = lambda st, lot: {
        "s_waiting": rec(RecordStatusEnum.waiting),
        "s_running": rec(RecordStatusEnum.running),
        "s_error": rec(RecordStatusEnum.error),
        "s_missing": None,
        "s_done": rec(RecordStatusEnum.complete),
        "s_fail": rec(RecordStatusEnum.complete),
    }[st]
    mock_stoich.return_value = {"be_nocp": [], "bsse": []}

    def add(client, base, name, stoich):
        if name == "s_fail":
            raise RuntimeError("server refused")
    mock_add.side_effect = add

    logger = MagicMock()
    opt_stru = {k: None for k in
                ("s_waiting", "s_running", "s_error", "s_missing", "s_done", "s_fail")}
    create_or_load_reaction_dataset(
        MagicMock(), "be_H2O_W5", "hf3c_minix", MagicMock(), MagicMock(),
        ds_opt, opt_stru, logger,
    )

    # be_stoichiometry only ever sees COMPLETE records (final_molecule set)
    assert mock_stoich.call_count == 2
    # n_entries counts only successful add_reaction calls
    msgs = [c.args[0] for c in logger.info.call_args_list]
    assert any("Created a total of 1 entries" in m for m in msgs)
    warns = " ".join(c.args[0] for c in logger.warning.call_args_list)
    assert "s_waiting" in warns and "s_running" in warns and "s_missing" in warns


# ---------------------------------------------------------------------------
# check_optimized_molecule — missing entry -> LevelOfTheoryNotFound
# ---------------------------------------------------------------------------

def test_check_optimized_molecule_missing_entry_raises_lot_not_found():
    from beep.adapters.qcfractal_adapter import (
        check_optimized_molecule, PortalRequestError,
    )
    from beep.core.errors import LevelOfTheoryNotFound

    ds = MagicMock()
    ds.name = "astro_mol"
    ds.get_record.side_effect = PortalRequestError(
        "Request failed: Missing 1 entries: XYZ (HTTP status 400)",
        status_code=400, details={},
    )
    with pytest.raises(LevelOfTheoryNotFound, match="does not exist"):
        check_optimized_molecule(ds, "hf3c_minix", "XYZ")


def test_check_optimized_molecule_missing_spec_raises_lot_not_found():
    from beep.adapters.qcfractal_adapter import check_optimized_molecule
    from beep.core.errors import LevelOfTheoryNotFound

    ds = MagicMock()
    ds.name = "astro_mol"
    ds.get_record.return_value = None
    with pytest.raises(LevelOfTheoryNotFound):
        check_optimized_molecule(ds, "hf3c_minix", ["XYZ"])


def test_check_optimized_molecule_propagates_server_errors():
    from beep.adapters.qcfractal_adapter import (
        check_optimized_molecule, PortalRequestError,
    )
    ds = MagicMock()
    ds.get_record.side_effect = PortalRequestError(
        "Internal server error", status_code=500, details={},
    )
    with pytest.raises(PortalRequestError):
        check_optimized_molecule(ds, "hf3c_minix", "XYZ")


# ---------------------------------------------------------------------------
# rmsd_filter_from_dataset — ERROR records warned, exact duplicates dropped
# ---------------------------------------------------------------------------

def _rmsd_ds(records):
    ds = MagicMock()
    ds.entry_names = list(records)
    ds.get_record.side_effect = lambda name, lot: records[name]
    return ds


def _opt_record(status, rmsd_to_others=1.0):
    r = MagicMock()
    r.status = status
    mol = MagicMock()
    mol.align.return_value = (None, {"rmsd": rmsd_to_others})
    r.final_molecule = mol
    return r


def test_rmsd_filter_warns_on_error_records():
    from beep.adapters.qcfractal_adapter import rmsd_filter_from_dataset

    ds = _rmsd_ds({
        "ok": _opt_record(RecordStatusEnum.complete),
        "bad": _opt_record(RecordStatusEnum.error),
    })
    logger = MagicMock()
    out = rmsd_filter_from_dataset(ds, "hf3c_minix", logger)
    assert list(out) == ["ok"]
    warns = " ".join(c.args[0] for c in logger.warning.call_args_list)
    assert "bad" in warns and "ERROR" in warns


def test_rmsd_filter_drops_exact_duplicates():
    """Two entries with identical geometry (RMSD == 0.0) are duplicates and
    the second must be dropped; the old ``rmsd != 0.0`` guard kept them."""
    from beep.adapters.qcfractal_adapter import rmsd_filter_from_dataset

    ds = _rmsd_ds({
        "a": _opt_record(RecordStatusEnum.complete, rmsd_to_others=0.0),
        "b": _opt_record(RecordStatusEnum.complete, rmsd_to_others=0.0),
    })
    out = rmsd_filter_from_dataset(ds, "hf3c_minix", MagicMock())
    assert list(out) == ["a"]
    # No self-comparison: only the (a, b) pair was aligned.
    assert ds.get_record.call_count == 2
    assert out["a"].align.call_count == 1


# ---------------------------------------------------------------------------
# _check_insert_meta / keyword type checks
# ---------------------------------------------------------------------------

def test_check_insert_meta_warns_on_failed_insert(caplog):
    from qcportal.metadata_models import InsertMetadata
    from beep.adapters.qcfractal_adapter import _check_insert_meta

    meta = InsertMetadata(
        errors=[(0, "Specification 'x' already exists with different content")],
        existing_idx=[],
    )
    logger = logging.getLogger("beep_insert_meta_test")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        ok = _check_insert_meta(meta, "specification 'x' in ds", logger)
    assert ok is False
    assert "did not fully accept specification 'x' in ds" in caplog.text
    assert "different content" in caplog.text


def test_check_insert_meta_silent_on_success(caplog):
    from qcportal.metadata_models import InsertMetadata
    from beep.adapters.qcfractal_adapter import _check_insert_meta

    logger = logging.getLogger("beep_insert_meta_test")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert _check_insert_meta(InsertMetadata(existing_idx=[0]), "spec", logger) is True
        assert _check_insert_meta(None, "spec", logger) is True
        assert _check_insert_meta(MagicMock(), "spec", logger) is True
    assert caplog.text == ""


def test_add_opt_specification_checks_insert_meta(caplog):
    from qcportal.metadata_models import InsertMetadata
    from beep.adapters.qcfractal_adapter import add_opt_specification

    ds = MagicMock()
    ds.name = "ds"
    ds.add_specification.return_value = InsertMetadata(errors=[(0, "conflict")])
    beep_logger = logging.getLogger("beep")
    beep_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="beep"):
            add_opt_specification(ds, {"name": "S", "qc_spec": {"method": "hf", "basis": "sto-3g"}})
    finally:
        beep_logger.removeHandler(caplog.handler)
    assert "optimization specification 's'" in caplog.text
    assert "conflict" in caplog.text


@pytest.mark.parametrize("bad", [5, "scf_type=df", ["scf_type", "df"]])
def test_add_opt_specification_rejects_non_dict_keywords(bad):
    """A legacy integer keyword ID (or any non-dict) used to be silently
    replaced by {}; it must now fail loudly."""
    from beep.adapters.qcfractal_adapter import add_opt_specification

    ds = MagicMock()
    with pytest.raises(TypeError, match="must be a dict"):
        add_opt_specification(ds, {
            "name": "s", "qc_spec": {"method": "hf", "basis": "sto-3g", "keywords": bad},
        })
    ds.add_specification.assert_not_called()


def test_add_opt_specification_rejects_non_dict_opt_keywords():
    from beep.adapters.qcfractal_adapter import add_opt_specification

    ds = MagicMock()
    with pytest.raises(TypeError, match="optimization keywords"):
        add_opt_specification(ds, {
            "name": "s", "qc_spec": {"method": "hf", "basis": "sto-3g"},
            "optimization_spec": {"keywords": 3},
        })


def test_add_opt_specification_passes_dict_keywords_through():
    from beep.adapters.qcfractal_adapter import add_opt_specification

    ds = MagicMock()
    add_opt_specification(ds, {
        "name": "s",
        "qc_spec": {"method": "hf", "basis": "sto-3g", "keywords": {"scf_type": "df"}},
        "optimization_spec": {"keywords": {"maxiter": 50}},
    })
    spec = ds.add_specification.call_args.args[1]
    assert spec.qc_specification.keywords == {"scf_type": "df"}
    assert spec.keywords == {"maxiter": 50}


def test_submit_energies_rejects_non_dict_keywords():
    from beep.adapters.qcfractal_adapter import submit_energies

    with pytest.raises(TypeError, match="must be a dict"):
        submit_energies(MagicMock(), "base", "pbe", "def2-svp", "psi4",
                        "be_nocp", "tag", keywords=42)


def test_submit_hessians_rejects_non_dict_keywords():
    from beep.adapters.qcfractal_adapter import submit_hessians

    with pytest.raises(TypeError, match="must be a dict"):
        submit_hessians(MagicMock(), "psi4", "hf", "sto-3g", [1], 42, "tag")
