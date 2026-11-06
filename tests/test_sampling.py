"""Tests for beep/core/sampling.py."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qcelemental.models.molecule import Molecule

from beep.core.sampling import (
    generate_shell_list,
    compute_rmsd_conditional,
    filter_binding_sites,
)


# ---------------------------------------------------------------------------
# generate_shell_list
# ---------------------------------------------------------------------------

def test_generate_shell_list_sparse():
    result = generate_shell_list(10.0, "sparse")
    assert result == [10.0]


def test_generate_shell_list_normal():
    result = generate_shell_list(10.0, "normal")
    assert len(result) == 3
    assert result == [10.0, 8.0, 12.0]


def test_generate_shell_list_fine():
    result = generate_shell_list(10.0, "fine")
    assert len(result) == 5
    assert result == [10.0, 8.0, 12.0, 7.5, 15.0]


def test_generate_shell_list_hyperfine():
    result = generate_shell_list(10.0, "hyperfine")
    assert len(result) == 7


def test_generate_shell_list_invalid():
    with pytest.raises(ValueError):
        generate_shell_list(10.0, "bogus")


# ---------------------------------------------------------------------------
# compute_rmsd_conditional
# ---------------------------------------------------------------------------

def test_compute_rmsd_identical(h2_mol):
    r, rm = compute_rmsd_conditional(h2_mol, h2_mol, rmsd_symm=False, cutoff=0.4)
    assert abs(r) < 1e-10


def test_compute_rmsd_no_mirror(h2_mol):
    r, rm = compute_rmsd_conditional(h2_mol, h2_mol, rmsd_symm=False, cutoff=0.4)
    assert rm == 10.0  # sentinel value when mirror not used


def test_compute_rmsd_with_mirror(h2_mol):
    # With rmsd_symm=True and a tight cutoff that r >= cutoff (shift the molecule)
    geom = np.array(h2_mol.geometry) + np.array([0.01, 0.0, 0.0])
    shifted = Molecule(symbols=h2_mol.symbols, geometry=geom.flatten())
    r, rm = compute_rmsd_conditional(h2_mol, shifted, rmsd_symm=True, cutoff=0.0001)
    # Mirror path should have been taken since r >= cutoff
    assert rm != 10.0 or r < 0.0001


# ---------------------------------------------------------------------------
# filter_binding_sites
# ---------------------------------------------------------------------------

def test_filter_empty_inputs(test_logger):
    result = filter_binding_sites(
        [], [], cut_off_val=0.4, rmsd_symm=False,
        logger=test_logger, ligand_size=2,
    )
    assert result == []


def test_filter_no_duplicates(h2_mol, ws3_cluster, test_logger):
    # Two distinct molecules — both should be kept
    geom_shifted = np.array(ws3_cluster.geometry) + np.array([100.0, 0.0, 0.0])
    # Create two structures: ws3+h2 at different positions
    symbols1 = list(ws3_cluster.symbols) + list(h2_mol.symbols)
    geom1 = np.concatenate([ws3_cluster.geometry, h2_mol.geometry]).flatten()
    mol1 = Molecule(symbols=symbols1, geometry=geom1)

    geom2_h2 = np.array(h2_mol.geometry) + np.array([100.0, 0.0, 0.0])
    geom2 = np.concatenate([ws3_cluster.geometry, geom2_h2]).flatten()
    mol2 = Molecule(symbols=symbols1, geometry=geom2)

    result = filter_binding_sites(
        [("a", mol1), ("b", mol2)], [],
        cut_off_val=0.01, rmsd_symm=False,
        logger=test_logger, ligand_size=len(h2_mol.symbols),
    )
    assert len(result) == 2


def test_filter_removes_duplicate(h2_mol, ws3_cluster, test_logger):
    # Two identical molecules — one should be removed
    symbols = list(ws3_cluster.symbols) + list(h2_mol.symbols)
    geom = np.concatenate([ws3_cluster.geometry, h2_mol.geometry]).flatten()
    mol = Molecule(symbols=symbols, geometry=geom)

    result = filter_binding_sites(
        [("a", mol), ("b", mol)], [],
        cut_off_val=0.4, rmsd_symm=False,
        logger=test_logger, ligand_size=len(h2_mol.symbols),
    )
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Real optimized structures from QCFractal (CO on w2/w3 clusters)
# ---------------------------------------------------------------------------

def test_rmsd_real_same_structure(co_w2_0001):
    """RMSD of a real binding site against itself should be ~0."""
    r, rm = compute_rmsd_conditional(co_w2_0001, co_w2_0001, rmsd_symm=False, cutoff=0.4)
    assert abs(r) < 1e-10


def test_rmsd_real_different_binding_sites(co_w2_0001, co_w2_0007):
    """Two different CO-w2 binding sites should have non-zero RMSD."""
    r, rm = compute_rmsd_conditional(co_w2_0001, co_w2_0007, rmsd_symm=False, cutoff=0.4)
    assert r > 0.01


def test_filter_real_distinct_sites_kept(co_w2_0001, co_w2_0007, test_logger):
    """Two genuinely different binding sites should both survive filtering."""
    ligand_size = 2  # CO has 2 atoms
    result = filter_binding_sites(
        [("co_w2_0001", co_w2_0001), ("co_w2_0007", co_w2_0007)], [],
        cut_off_val=0.25, rmsd_symm=False,
        logger=test_logger, ligand_size=ligand_size,
    )
    assert len(result) == 2


def test_filter_real_against_reference(co_w2_0001, co_w3_0001, co_w3_0004, test_logger):
    """Filter new candidates against an existing reference set."""
    ligand_size = 2  # CO
    # co_w3 structures are different systems (3 waters vs 2) so they won't match
    result = filter_binding_sites(
        [("co_w3_0001", co_w3_0001), ("co_w3_0004", co_w3_0004)],
        [("co_w2_0001", co_w2_0001)],
        cut_off_val=0.25, rmsd_symm=False,
        logger=test_logger, ligand_size=ligand_size,
        atoms_map=False,  # different atom counts, can't use atoms_map
    )
    # w3 structures have 11 atoms vs w2's 8 — can't align, so both should survive
    assert len(result) >= 1


def test_rmsd_real_co_w5_different_sites(co_w5_0001, co_w5_0002):
    """Two different CO-w5 binding sites should have non-zero RMSD."""
    r, rm = compute_rmsd_conditional(co_w5_0001, co_w5_0002, rmsd_symm=False, cutoff=0.4)
    assert r > 0.01


def test_filter_real_co_w5_distinct_kept(co_w5_0001, co_w5_0002, test_logger):
    """Two distinct CO-w5 binding sites should both survive filtering."""
    ligand_size = 2  # CO
    result = filter_binding_sites(
        [("co_w5_0001", co_w5_0001), ("co_w5_0002", co_w5_0002)], [],
        cut_off_val=0.25, rmsd_symm=False,
        logger=test_logger, ligand_size=ligand_size,
    )
    assert len(result) == 2


# ---------------------------------------------------------------------------
# run_sampling — workflow-level case-B regression
# ---------------------------------------------------------------------------

def test_run_sampling_submits_at_new_lot_when_all_entries_exist(test_logger):
    """Regression for reports/BUG_beep_sampling_spec_change.md.

    When the user re-runs sampling at a different LOT against a dataset
    whose entries already exist (from a prior LOT), the workflow must
    still call submit_optimizations at the current LOT for those
    pre-existing entries. The old code only submitted when there were
    *new* entries to add, so a LOT change on already-populated datasets
    silently produced zero new opts.
    """
    from beep.workflows.sampling import run_sampling

    cluster_name = "CO_W3_01"
    # Names that the workflow will construct for max_structures=3
    existing_entry_names = [f"{cluster_name}_{i:04d}" for i in (1, 2, 3)]

    sampling_dset = MagicMock()
    sampling_dset.name = f"pre_{cluster_name}"
    sampling_dset.entry_names = existing_entry_names
    # iterate_entries yields nothing — keeps the "existing molecules" list empty
    refinement_dset = MagicMock()
    refinement_dset.name = cluster_name
    refinement_dset.iterate_entries.return_value = iter([])
    refinement_dset.entry_names = []

    cluster_mol = MagicMock(); cluster_mol.symbols = ["O", "H", "H"]
    target_mol = MagicMock(); target_mol.symbols = ["C", "O"]

    client = MagicMock()

    submit_call_kwargs = []

    def fake_submit_optimizations(ds_opt, opt_lot, tag, subset=None):
        submit_call_kwargs.append(
            {"ds": ds_opt.name, "opt_lot": opt_lot, "tag": tag,
             "subset": list(subset) if subset else None}
        )
        r = MagicMock(); r.n_inserted = len(subset or []); r.n_existing = 0
        return r

    with patch("beep.workflows.sampling.qcf") as mock_qcf:
        mock_qcf.add_opt_specification.return_value = None
        mock_qcf.get_job_ids.return_value = []
        mock_qcf.wait_for_completion.return_value = None
        mock_qcf.submit_optimizations.side_effect = fake_submit_optimizations
        mock_qcf.fetch_opt_molecules.return_value = []
        mock_qcf.add_opt_entry.return_value = None

        # generate_shell_list("sparse", 2.0) returns a single shell, so the
        # per-shell loop runs exactly once.
        run_sampling(
            method="gfn2-xtb", basis=None, program="xtb",
            tag="sampling", kw_id=None,
            sampling_opt_dset=sampling_dset,
            refinement_opt_dset=refinement_dset,
            opt_lot="gfn2-xtb",
            rmsd_symm=False, store_initial=False, rmsd_val=0.4,
            target_mol=target_mol, cluster=cluster_mol,
            debug_path="/tmp/dbg",
            client=client,
            sampling_shell=2.0, sampling_condition="sparse",
            logger=test_logger,
        )

    # The fix: submit_optimizations MUST be called for the pre-existing
    # entries at the new opt_lot, even though no new entries were added.
    assert len(submit_call_kwargs) == 1, submit_call_kwargs
    call = submit_call_kwargs[0]
    assert call["opt_lot"] == "gfn2-xtb"
    assert call["tag"] == "sampling"
    assert sorted(call["subset"]) == sorted(existing_entry_names)


# ---------------------------------------------------------------------------
# run_sampling / run — option handling regressions
# ---------------------------------------------------------------------------

def _mock_dsets(cluster_name, existing=()):
    sampling_dset = MagicMock()
    sampling_dset.name = f"pre_{cluster_name}"
    sampling_dset.entry_names = list(existing)
    refinement_dset = MagicMock()
    refinement_dset.name = cluster_name
    refinement_dset.iterate_entries.return_value = iter([])
    refinement_dset.entry_names = []
    return sampling_dset, refinement_dset


def test_store_initial_filename_keeps_dotted_directories(tmp_path, ws3_cluster, h2_mol, test_logger):
    """Regression: .replace('.', '') ran on the whole path, so a cwd such as
    ~/beep-0.12/ or ~/.local/ was mangled into a non-existent directory."""
    from beep.workflows.sampling import run_sampling

    dotted_dir = tmp_path / "beep-0.12" / "site_finder"
    dotted_dir.mkdir(parents=True)
    debug_path = dotted_dir / "CO_W3_01"
    sampling_dset, refinement_dset = _mock_dsets("CO_W3_01")

    debug_mol = MagicMock()
    with patch("beep.workflows.sampling.qcf") as mock_qcf, \
         patch("beep.core.molecule_sampler.random_molecule_sampler",
               return_value=([], debug_mol)):
        mock_qcf.get_job_ids.return_value = []
        mock_qcf.fetch_opt_molecules.return_value = []
        run_sampling(
            method="gfn2-xtb", basis=None, program="xtb", tag="sampling", kw_id=None,
            sampling_opt_dset=sampling_dset, refinement_opt_dset=refinement_dset,
            opt_lot="gfn2-xtb", rmsd_symm=False, store_initial=True, rmsd_val=0.4,
            target_mol=h2_mol, cluster=ws3_cluster, debug_path=debug_path,
            client=MagicMock(), sampling_shell=2.0, sampling_condition="sparse",
            logger=test_logger,
        )

    debug_mol.to_file.assert_called_once()
    written = debug_mol.to_file.call_args[0][0]
    from pathlib import Path
    assert Path(written).parent == dotted_dir
    assert Path(written).name == "CO_W3_01_200.mol"


def test_anchor_fallback_logs_warning():
    """The n_water//3 fallback used to be a silent bare except; the user must see
    a warning naming the exception."""
    from beep.workflows.sampling import run_sampling

    sampling_dset, refinement_dset = _mock_dsets(
        "CO_W3_01", existing=[f"CO_W3_01_{i:04d}" for i in (1, 2, 3)]
    )
    cluster_mol = MagicMock(); cluster_mol.symbols = ["O", "H", "H"]
    target_mol = MagicMock(); target_mol.symbols = ["C", "O"]
    logger = MagicMock()
    with patch("beep.workflows.sampling.qcf") as mock_qcf, \
         patch("beep.core.molecule_sampler.adaptive_shift_vectors",
               side_effect=RuntimeError("degenerate geometry")):
        mock_qcf.get_job_ids.return_value = []
        mock_qcf.fetch_opt_molecules.return_value = []
        run_sampling(
            method="gfn2-xtb", basis=None, program="xtb", tag="sampling", kw_id=None,
            sampling_opt_dset=sampling_dset, refinement_opt_dset=refinement_dset,
            opt_lot="gfn2-xtb", rmsd_symm=False, store_initial=False, rmsd_val=0.4,
            target_mol=target_mol, cluster=cluster_mol, debug_path="/tmp/dbg",
            client=MagicMock(), sampling_shell=2.0, sampling_condition="sparse",
            logger=logger,
        )
    warnings = [c for c in logger.warning.call_args_list if "falling back" in str(c.args[0])]
    assert len(warnings) == 1
    # the exception type/message is part of the warning, not swallowed
    rendered = warnings[0].args[0] % warnings[0].args[1:]
    assert "RuntimeError" in rendered and "degenerate geometry" in rendered


def test_sampling_config_qc_keywords_accepts_dict_and_rejects_legacy_keyword_id():
    """`qc_keywords` takes the QC-program keywords as a dict. The old
    `keyword_id` name still loads when null but a legacy QCFractal keyword
    ID (int/str) or a dict under that name is rejected pointing to
    `qc_keywords`, instead of being warned about and ignored at run time."""
    from pydantic import ValidationError
    from beep.models.sampling import SamplingConfig
    base = dict(
        workflow="sampling", molecule="CO", total_binding_sites=1,
        sampling_level_of_theory={"method": "gfn2-xtb", "program": "xtb"},
        refinement_level_of_theory={"method": "hf", "basis": "sto-3g", "program": "psi4"},
    )
    cfg = SamplingConfig(**base, qc_keywords={"guess": "gwh"})
    assert cfg.qc_keywords == {"guess": "gwh"}
    assert SamplingConfig(**base, qc_keywords=None).qc_keywords is None
    assert SamplingConfig(**base, keyword_id=None).keyword_id is None
    for legacy in (42, "legacy", {"guess": "gwh"}):
        with pytest.raises(ValidationError, match="qc_keywords"):
            SamplingConfig(**base, keyword_id=legacy)


def test_run_passes_keywords_to_refinement_spec(tmp_path, monkeypatch):
    """Regression: the user's QC keywords were silently discarded on the way to
    add_opt_specification. They must arrive verbatim as the refinement qc_spec
    keywords, and never leak into the sampling spec."""
    from beep.models.sampling import SamplingConfig
    from beep.workflows import sampling as wf

    monkeypatch.chdir(tmp_path)
    cfg = SamplingConfig(
        workflow="sampling", molecule="CO", total_binding_sites=1,
        sampling_level_of_theory={"method": "gfn2-xtb", "program": "xtb"},
        refinement_level_of_theory={"method": "hf", "basis": "sto-3g", "program": "psi4"},
        qc_keywords={"guess": "gwh", "damping_percentage": 20},
    )
    ds_wc = MagicMock(); ds_wc.entry_names = ["W3_01", "W3_02"]
    ds_sm = MagicMock()
    dsets = _empty_datasets(["W3_01", "W3_02"])
    target = MagicMock(); target.symbols = ["C", "O"]
    specs = []

    def fake_sampling(**kw):
        kw["refinement_opt_dset"].entry_names = ["CO_W3_01_0001"]

    with patch.object(wf, "qcf") as qcf, \
         patch.object(wf, "run_sampling", side_effect=fake_sampling) as run_sampling:
        qcf.get_collection.side_effect = lambda c, k, n: ds_sm if n == cfg.small_molecule_collection else ds_wc
        qcf.fetch_initial_molecule.return_value = target
        qcf.get_or_create_opt_dataset.side_effect = lambda client, name: dsets[name]
        qcf.add_opt_specification.side_effect = lambda ds, spec, overwrite=True: specs.append(spec)
        qcf.get_job_ids.return_value = []
        wf.run(cfg, MagicMock())

    assert len(specs) == 1
    assert specs[0]["name"] == cfg.refinement_level_of_theory.lot_name
    assert specs[0]["qc_spec"]["keywords"] == {"guess": "gwh", "damping_percentage": 20}
    assert run_sampling.call_args.kwargs["kw_id"] is None
    # total_binding_sites=1 reached after the first cluster (>=, not >): stop early
    assert run_sampling.call_count == 1


def _empty_datasets(clusters, mol="CO"):
    """One MagicMock per '<mol>_<w>' / 'pre_<mol>_<w>' dataset, all empty."""
    dsets = {}
    for w in clusters:
        for name in (f"{mol}_{w}", f"pre_{mol}_{w}"):
            ds = MagicMock(); ds.name = name; ds.entry_names = []
            dsets[name] = ds
    return dsets


def _resume_config(total):
    from beep.models.sampling import SamplingConfig
    return SamplingConfig(
        workflow="sampling", molecule="CO", total_binding_sites=total,
        sampling_level_of_theory={"method": "gfn2-xtb", "program": "xtb"},
        refinement_level_of_theory={"method": "hf", "basis": "sto-3g", "program": "psi4"},
    )


def _run_with_mocks(cfg, dsets, clusters, harvest_side_effect=None, sampling_side_effect=None):
    from beep.workflows import sampling as wf

    ds_wc = MagicMock(); ds_wc.entry_names = list(clusters)
    ds_sm = MagicMock()
    target = MagicMock(); target.symbols = ["C", "O"]
    with patch.object(wf, "qcf") as qcf, \
         patch.object(wf, "run_sampling", side_effect=sampling_side_effect) as run_sampling, \
         patch.object(wf, "harvest_cluster", side_effect=harvest_side_effect) as harvest, \
         patch.object(wf, "process_refinement") as refine:
        qcf.get_collection.side_effect = lambda c, k, n: ds_sm if n == cfg.small_molecule_collection else ds_wc
        qcf.fetch_initial_molecule.return_value = target
        qcf.get_or_create_opt_dataset.side_effect = lambda client, name: dsets[name]
        qcf.get_job_ids.side_effect = lambda ds, entries, lot: list(range(len(entries)))
        qcf.check_for_completion.return_value = (True, {"COMPLETE": 0})
        wf.run(cfg, MagicMock())
    return run_sampling, harvest, refine


def test_run_relaunch_does_not_resample_when_target_already_met(tmp_path, monkeypatch):
    """Regression: a relaunch after a crash re-sampled every finished cluster,
    paying a full gfn2 round per cluster before noticing the target was met.
    With enough refined sites already stored, run() must not place anything."""
    monkeypatch.chdir(tmp_path)
    clusters = ["W3_01", "W3_02"]
    dsets = _empty_datasets(clusters)
    dsets["CO_W3_01"].entry_names = ["CO_W3_01_0001", "CO_W3_01_0002"]
    dsets["pre_CO_W3_01"].entry_names = [f"pre_{i}" for i in range(5)]
    dsets["CO_W3_02"].entry_names = ["CO_W3_02_0001"]
    dsets["pre_CO_W3_02"].entry_names = [f"pre_{i}" for i in range(5)]

    run_sampling, harvest, refine = _run_with_mocks(
        _resume_config(total=3), dsets, clusters, harvest_side_effect=lambda *a, **k: 0,
    )

    assert run_sampling.call_count == 0
    # every cluster with sampling entries is harvested, even the ones after the target
    assert harvest.call_count == 2
    # Ensure calculations exist even when harvesting adds no new geometries.
    assert refine.call_count == 2


@pytest.mark.parametrize("has_sampling_entries", [False, True])
@pytest.mark.parametrize("scenario", ["interrupted_submission", "new_level", "already_complete"])
def test_resume_ensures_refinement_records_before_finishing(
    tmp_path, monkeypatch, has_sampling_entries, scenario,
):
    from beep.workflows import sampling as wf

    monkeypatch.chdir(tmp_path)
    cfg = _resume_config(total=1)
    old_lot = cfg.refinement_level_of_theory.lot_name
    if scenario == "new_level":
        cfg.refinement_level_of_theory = cfg.refinement_level_of_theory.model_copy(
            update={"method": "b3lyp"},
        )
    requested_lot = cfg.refinement_level_of_theory.lot_name
    dsets = _empty_datasets(["W3_01", "W3_02"])
    ds_ref = dsets["CO_W3_01"]
    ds_ref.entry_names = ["CO_W3_01_0001"]
    if has_sampling_entries:
        dsets["pre_CO_W3_01"].entry_names = list(ds_ref.entry_names)
    records = {} if scenario == "interrupted_submission" else {old_lot: [101]}
    original_records = dict(records)
    inserted = []

    def submit(ds, lot, tag):
        assert ds is ds_ref
        assert lot == requested_lot
        if lot not in records:
            records[lot] = [102]
            inserted.append(102)
        return MagicMock(n_inserted=len(inserted), n_existing=int(not inserted))

    surface = MagicMock()
    surface.entry_names = ["W3_01", "W3_02"]
    target = MagicMock()
    target.symbols = ["C", "O"]
    with patch.object(wf, "qcf") as qcf, patch.object(wf, "run_sampling") as sample:
        qcf.get_collection.return_value = surface
        qcf.fetch_initial_molecule.return_value = target
        qcf.get_or_create_opt_dataset.side_effect = lambda client, name: dsets[name]
        qcf.get_job_ids.side_effect = lambda ds, entries, lot: records.get(lot, []) if ds is ds_ref else []
        qcf.submit_optimizations.side_effect = submit
        qcf.check_for_completion.return_value = (True, {"COMPLETE": 1})

        wf.run(cfg, MagicMock())

    sample.assert_not_called()
    qcf.submit_optimizations.assert_called_once_with(ds_ref, requested_lot, tag=cfg.refinement_tag)
    qcf.add_opt_specification.assert_called_once()
    spec = qcf.add_opt_specification.call_args.args[1]
    assert spec["name"] == requested_lot
    assert spec["qc_spec"]["method"] == cfg.refinement_level_of_theory.qc_method
    assert len(inserted) == (0 if scenario == "already_complete" else 1)
    for lot, ids in original_records.items():
        assert records[lot] == ids
    qcf.wait_for_completion.assert_called_once()
    assert qcf.wait_for_completion.call_args.args[1] == records[requested_lot]
    qcf.reset_records.assert_not_called()


def test_run_harvest_counts_before_top_up(tmp_path, monkeypatch):
    """Harvested sites count toward the target: with 1 stored and 2 harvested
    (recovered after a crash) against a target of 3, no new placements."""
    monkeypatch.chdir(tmp_path)
    clusters = ["W3_01", "W3_02"]
    dsets = _empty_datasets(clusters)
    dsets["CO_W3_01"].entry_names = ["CO_W3_01_0001"]
    dsets["pre_CO_W3_01"].entry_names = [f"pre_{i}" for i in range(5)]

    def fake_harvest(client, ds_smplg, ds_ref, *a, **k):
        ds_ref.entry_names = ds_ref.entry_names + ["CO_W3_01_0002", "CO_W3_01_0003"]
        return 2

    run_sampling, harvest, refine = _run_with_mocks(
        _resume_config(total=3), dsets, clusters, harvest_side_effect=fake_harvest,
    )

    assert harvest.call_count == 1          # W3_02 has no sampling entries: skipped
    assert refine.call_count == 1           # new sites -> refinement resubmitted
    assert run_sampling.call_count == 0


def test_run_tops_up_only_while_below_target(tmp_path, monkeypatch):
    """Below target after harvesting, run() falls back to new placements and
    stops as soon as the running total reaches the target."""
    monkeypatch.chdir(tmp_path)
    clusters = ["W3_01", "W3_02", "W3_03"]
    dsets = _empty_datasets(clusters)
    dsets["CO_W3_01"].entry_names = ["CO_W3_01_0001"]
    dsets["pre_CO_W3_01"].entry_names = ["pre_0"]

    def fake_sampling(**kw):
        ds_ref = kw["refinement_opt_dset"]
        ds_ref.entry_names = ds_ref.entry_names + [f"{ds_ref.name}_new1", f"{ds_ref.name}_new2"]

    run_sampling, harvest, refine = _run_with_mocks(
        _resume_config(total=4), dsets, clusters,
        harvest_side_effect=lambda *a, **k: 0, sampling_side_effect=fake_sampling,
    )

    # fresh clusters first: 1 stored + 2 (W3_02) = 3 < 4, + 2 (W3_03) = 5 >= 4
    # -> the already-sampled W3_01 is never topped up
    assert run_sampling.call_count == 2
    sampled = [c.kwargs["sampling_opt_dset"].name for c in run_sampling.call_args_list]
    assert sampled == ["pre_CO_W3_02", "pre_CO_W3_03"]
    assert refine.call_count == 3  # stored W3_01 entries, then the two fresh clusters


def test_run_top_up_visits_never_sampled_clusters_first(tmp_path, monkeypatch):
    """A relaunch below target must place new sites on clusters that were
    never sampled before topping up ones that already had a round, even when
    the sampled cluster comes first in the surface collection."""
    monkeypatch.chdir(tmp_path)
    clusters = ["W3_01", "W3_02", "W3_03"]
    dsets = _empty_datasets(clusters)
    dsets["CO_W3_01"].entry_names = ["CO_W3_01_0001"]
    dsets["pre_CO_W3_01"].entry_names = [f"pre_{i}" for i in range(5)]

    def fake_sampling(**kw):
        ds_ref = kw["refinement_opt_dset"]
        ds_ref.entry_names = ds_ref.entry_names + [f"{ds_ref.name}_new1", f"{ds_ref.name}_new2"]

    run_sampling, harvest, refine = _run_with_mocks(
        _resume_config(total=3), dsets, clusters,
        harvest_side_effect=lambda *a, **k: 0, sampling_side_effect=fake_sampling,
    )

    # 1 stored + 2 from the first fresh cluster = 3 >= 3: W3_01 is never topped up
    sampled = [c.kwargs["sampling_opt_dset"].name for c in run_sampling.call_args_list]
    assert sampled == ["pre_CO_W3_02"]


def test_run_top_up_falls_back_to_sampled_clusters_after_fresh_ones(tmp_path, monkeypatch):
    """Sampled clusters are topped up only once every fresh cluster had a round."""
    monkeypatch.chdir(tmp_path)
    clusters = ["W3_01", "W3_02"]
    dsets = _empty_datasets(clusters)
    dsets["pre_CO_W3_01"].entry_names = ["pre_0"]

    def fake_sampling(**kw):
        ds_ref = kw["refinement_opt_dset"]
        ds_ref.entry_names = ds_ref.entry_names + [f"{ds_ref.name}_new"]

    run_sampling, harvest, refine = _run_with_mocks(
        _resume_config(total=2), dsets, clusters,
        harvest_side_effect=lambda *a, **k: 0, sampling_side_effect=fake_sampling,
    )

    sampled = [c.kwargs["sampling_opt_dset"].name for c in run_sampling.call_args_list]
    assert sampled == ["pre_CO_W3_02", "pre_CO_W3_01"]


def test_harvest_cluster_skips_entries_already_refined(test_logger):
    """harvest_cluster only filters sampling entries not yet promoted to the
    refinement dataset, and waits for the pending ones first."""
    from beep.workflows import sampling as wf

    ds_smplg = MagicMock(); ds_smplg.entry_names = ["a", "b", "c"]
    ds_ref = MagicMock(); ds_ref.entry_names = ["a"]
    target = MagicMock(); target.symbols = ["C", "O"]

    with patch.object(wf, "qcf") as qcf, \
         patch.object(wf, "_filter_and_register", return_value=1) as reg:
        qcf.get_job_ids.return_value = [1, 2, 3]
        qcf.fetch_opt_molecules.return_value = [("b", MagicMock())]
        n = wf.harvest_cluster(MagicMock(), ds_smplg, ds_ref, "lot", target, 0.4, False, test_logger)

    assert n == 1
    qcf.wait_for_completion.assert_called_once()
    assert qcf.fetch_opt_molecules.call_args.args[1] == ["b", "c"]
    assert qcf.fetch_opt_molecules.call_args.kwargs["status"] == "COMPLETE"
    assert reg.call_count == 1
    assert [name for name, _ in reg.call_args.args[0]] == ["b"]
