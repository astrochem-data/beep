"""Trajectory-benchmark orchestration helpers.

Workflow-level glue for the ``geom_benchmark`` trajectory analysis: for
each DFT functional, evaluate SP + gradient at every geometry along the
reference optimization trajectory and report MAE/RMSE of energy
(meV/atom) and forces (meV/Å) against the reference, plus a combined
z-score-weighted ranking.

The pure-math layer lives in :mod:`beep.core.trajectory_metrics`
(``per_step_deltas``, ``summarize_method_metrics``,
``combined_zscore_ranking``); the QCFractal I/O lives in
:mod:`beep.adapters.qcfractal_adapter`. This module orchestrates the
two against a benchmark's reference + DFT optimizations.

Inspired by Bovolenta et al. 2025 (arXiv 2508.14219), Appendix C.2 /
Fig C.1.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import is_complete, is_incomplete, is_error
from .logging_utils import (
    padded_log,
    log_trajectory_metrics_per_group,
    log_trajectory_ranking_table,
    log_nm_force_decomposition,
)
from .plotting_utils import trajectory_error_histograms
from .trajectory_metrics import (
    per_step_deltas, summarize_method_metrics, combined_zscore_ranking,
    decompose_force_errors_rigid, summarize_force_decomposition,
    FORCE_PART_KEYS,
)
from .dft_functionals import is_nl_method

BOHR_TO_ANGSTROM = 0.529177210903

# Config ranking_metric -> metrics key that enters the ranking as rmsd_force.
RANKING_FORCE_KEYS = {
    "total": "rmsd_force",
    "intermolecular": "rmsd_force_intermolecular",
}

bcheck = "✔"

TRAJECTORY_DECOMPOSITION_COLS = [
    ("rmsd_force_intermolecular", "Net force", "{:>10.2f}"),
    ("rmsd_torque", "Torque", "{:>10.2f}"),
    ("rmsd_force_intramolecular", "Intramol.", "{:>10.2f}"),
    ("rmsd_force", "Total", "{:>10.2f}"),
]


def _kabsch_rmsd(P, Q):
    """Minimal RMSD between two (n,3) coordinate sets (same units) after
    optimal rigid translation + rotation (Kabsch). Unweighted."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1.0, 1.0, d]) @ Wt
    Pc_aligned = Pc @ U
    return float(np.sqrt(np.mean(np.sum((Pc_aligned - Qc) ** 2, axis=1))))


def _dedup_trajectory_steps(steps, rmsd_thr, min_frames, logger, struct_name):
    """Group reference-trajectory frames by Kabsch-aligned RMSD and keep only
    the geometrically distinct ones (compute-saving pre-filter).

    Keeps the first frame, then each later frame whose RMSD to the last kept
    frame exceeds ``rmsd_thr`` (Angstrom). If fewer than ``min_frames`` survive
    (a near-stationary system), the most-displaced dropped frames are added
    back up to the floor. Returns the kept steps in trajectory order.
    """
    if not rmsd_thr or rmsd_thr <= 0 or len(steps) <= 1:
        return steps

    def coords_ang(st):
        g = np.asarray(st["molecule"].geometry, dtype=float).reshape(-1, 3)
        return g * BOHR_TO_ANGSTROM

    kept_idx = [0]
    last = coords_ang(steps[0])
    for i in range(1, len(steps)):
        c = coords_ang(steps[i])
        if _kabsch_rmsd(c, last) > rmsd_thr:
            kept_idx.append(i)
            last = c

    if len(kept_idx) < min_frames:
        ref = coords_ang(steps[0])
        dropped = [i for i in range(len(steps)) if i not in set(kept_idx)]
        dropped.sort(key=lambda i: _kabsch_rmsd(coords_ang(steps[i]), ref),
                     reverse=True)
        for i in dropped:
            if len(kept_idx) >= min_frames:
                break
            kept_idx.append(i)
        kept_idx.sort()

    if len(kept_idx) < len(steps):
        logger.info(
            f"  {struct_name}: dedup {len(steps)} -> {len(kept_idx)} frames "
            f"(RMSD threshold {rmsd_thr} Angstrom)"
        )
    return [steps[i] for i in kept_idx]


def collect_reference_trajectories(
    odset_dict: dict, geom_ref_opt_lot: str, logger: logging.Logger,
    dedup_rmsd: float = None, dedup_min_frames: int = 1,
) -> dict:
    """Pull each system's reference optimization trajectory.

    A fetch failure on one system (e.g. qcportal cache OverflowError on
    a record with corrupted property values) is logged with a prominent
    warning and the system is skipped — other systems still get analysed.
    Matches the defense-in-depth pattern of skipping a single bad record
    rather than aborting the whole workflow.
    """
    out = {}
    for struct_name, odset in odset_dict.items():
        try:
            steps = qcf.get_optimization_trajectory(
                odset, struct_name, geom_ref_opt_lot,
            )
        except Exception as exc:
            logger.warning("")
            logger.warning(
                f"  !! WARNING: {struct_name}: trajectory fetch failed "
                f"({type(exc).__name__}: {exc})."
            )
            logger.warning(
                f"  !!          This system will be EXCLUDED from the "
                f"trajectory benchmark; eq-geometry RMSD is unaffected."
            )
            logger.warning("")
            continue
        if not steps:
            logger.warning(
                f"  No reference trajectory for {struct_name} at "
                f"{geom_ref_opt_lot} — skipping system."
            )
            continue
        # Compute-saving dedup, before the gradient-coverage check so a
        # stationary gradient-less tail frame is collapsed away rather than
        # excluding the whole system.
        steps = _dedup_trajectory_steps(
            steps, dedup_rmsd, dedup_min_frames, logger, struct_name,
        )
        if any(s["gradient_hartree_per_bohr"] is None for s in steps):
            logger.warning(
                f"  Reference trajectory for {struct_name} missing gradient "
                f"on some steps — skipping system."
            )
            continue
        out[struct_name] = steps
        logger.info(
            f"  {struct_name}: {len(steps)} reference trajectory steps."
        )
    return out


def build_trajectory_sp_datasets(
    client, traj_data: dict, all_dft_functionals: List[str],
    program: str, dft_keyword, logger: logging.Logger,
) -> dict:
    """Per system: create ``<system>_trajref`` SinglepointDataset, add one
    entry per traj step (``step_NNN``), and register one SP+gradient spec
    per DFT functional. Idempotent — safe to re-run."""
    traj_dsets = {}
    # Inline keyword dict validated by the config model; pass it through.
    kw = dict(dft_keyword or {})
    for struct_name, steps in traj_data.items():
        ds_name = f"{struct_name}_trajref"
        ds_sp = qcf.get_or_create_singlepoint_dataset(client, ds_name)

        entries = [
            (f"step_{i:03d}", step["molecule"])
            for i, step in enumerate(steps)
        ]
        qcf.add_singlepoint_entries(ds_sp, entries)

        for functional in all_dft_functionals:
            method, basis = functional.split("_", 1)
            # VV10 "-V" methods need ORCA+SCNL for a complete gradient;
            # everything else uses the configured program.
            if is_nl_method(method):
                spec_program, spec_kw = "orca", {"simple_input": "SCNL"}
            else:
                spec_program, spec_kw = program, kw
            qcf.add_gradient_spec(
                ds_sp,
                spec_name=functional,
                method=method,
                basis=basis,
                program=spec_program,
                keywords=spec_kw,
                description=(
                    f"SP+gradient on ref trajectory geometry "
                    f"for {functional}"
                ),
            )
        traj_dsets[struct_name] = ds_sp
        logger.info(
            f"  {struct_name}: dataset '{ds_name}' "
            f"({len(entries)} entries, {len(all_dft_functionals)} specs)"
        )
    return traj_dsets


def submit_trajectory_singlepoints(
    traj_dsets: dict, all_dft_functionals: List[str],
    tag: str, logger: logging.Logger, tag_nl: str = None,
):
    """Submit all (entry × spec) SP+gradient pairs.

    Assumes ``all_dft_functionals`` is already lowercased (per the
    convention applied at the top of ``run_trajectory_analysis``,
    matching be_hess / energy_benchmark / sampling).

    VV10 "-V" specs (``is_nl_method``) are ORCA/SCNL jobs and go to
    ``tag_nl`` (an ORCA manager's tag); everything else goes to ``tag``.
    """
    nl_specs = [s for s in all_dft_functionals if is_nl_method(s.split("_", 1)[0])]
    std_specs = [s for s in all_dft_functionals if s not in nl_specs]
    groups = [(std_specs, tag)]
    if nl_specs:
        groups.append((nl_specs, tag_nl or tag))
    inserted = existing = 0
    for specs, grp_tag in groups:
        if not specs:
            continue
        for ds_sp in traj_dsets.values():
            meta = qcf.submit_singlepoints_in_dataset(
                ds_sp, spec_names=specs, tag=grp_tag,
            )
            inserted += getattr(meta, "n_inserted", 0)
            existing += getattr(meta, "n_existing", 0)
        logger.info(
            f"  Submitted {len(specs)} spec(s) across "
            f"{len(traj_dsets)} system(s) to tag '{grp_tag}'."
        )
    logger.info(
        f"  Total: {inserted} new SP+gradient computations "
        f"({existing} already existed)."
    )
    return inserted, existing


def wait_for_trajectory_completion(
    traj_dsets: dict, all_dft_functionals: List[str],
    wait_interval: int, logger: logging.Logger,
):
    """Poll until every (system × entry × spec) trajectory SP is terminal.

    Assumes ``all_dft_functionals`` is already lowercased.
    """
    while True:
        complete = incomplete = error = 0
        for ds_sp in traj_dsets.values():
            for entry_name in ds_sp.entry_names:
                for spec_key in all_dft_functionals:
                    record = ds_sp.get_record(entry_name, spec_key)
                    if record is None:
                        continue
                    if is_complete(record.status):
                        complete += 1
                    elif is_incomplete(record.status):
                        incomplete += 1
                    elif is_error(record.status):
                        error += 1
        if incomplete == 0:
            logger.info(
                f"  Traj SP: Complete: {complete}, Error: {error} {bcheck}"
            )
            return complete, error
        logger.info(
            f"  Traj SP: {complete} done, {incomplete} running, {error} err"
        )
        logger.info(f"  Waiting {wait_interval}s before rechecking...")
        time.sleep(wait_interval)


def compute_per_method_trajectory_metrics(
    traj_data: dict, traj_dsets: dict, all_dft_functionals: List[str],
    logger: logging.Logger, fragments_per_struct: dict = None,
):
    """For each functional, aggregate per-step deltas across systems and
    return ``(metrics, raw_deltas)``.

    ``metrics`` is a dict ``{functional → summarised MAE/RMSE numbers}``;
    ``raw_deltas`` is a dict ``{functional → {delta_e arrays, delta_force
    arrays}}`` for downstream plotting. Skips functionals with no
    complete records.

    With ``fragments_per_struct`` (``{structure → [[atom idx, ...], ...]}``)
    the force error of every step is also split per fragment by
    :func:`decompose_force_errors_rigid`, adding
    ``rmsd_force_intermolecular`` (net force), ``rmsd_torque`` and
    ``rmsd_force_intramolecular`` to the metrics and the matching flat
    arrays to ``raw_deltas``. A structure whose partition does not match
    its atom count is reported once and left out of the decomposition.
    """
    metrics: Dict[str, dict] = {}
    raw_deltas: Dict[str, dict] = {}
    fragments_per_struct = fragments_per_struct or {}
    bad_partition = set()

    for functional in all_dft_functionals:
        per_system = []
        per_system_parts = []
        skipped_systems = 0

        for struct_name, steps in traj_data.items():
            ds_sp = traj_dsets.get(struct_name)
            if ds_sp is None:
                continue

            ref_e, ref_g, dft_e, dft_g, pos, masses = [], [], [], [], [], None
            n_atoms = None
            for i, step in enumerate(steps):
                entry_name = f"step_{i:03d}"
                de, dg = qcf.fetch_sp_energy_gradient(
                    ds_sp, entry_name, functional,
                )
                if de is None or dg is None:
                    continue
                if (step["energy_hartree"] is None
                        or step["gradient_hartree_per_bohr"] is None):
                    continue
                if n_atoms is None:
                    n_atoms = len(step["molecule"].symbols)
                    masses = np.asarray(step["molecule"].masses, dtype=float)
                ref_e.append(step["energy_hartree"])
                ref_g.append(step["gradient_hartree_per_bohr"])
                dft_e.append(de)
                dft_g.append(dg)
                pos.append(
                    np.asarray(step["molecule"].geometry, dtype=float).reshape(-1, 3)
                    * BOHR_TO_ANGSTROM
                )

            if not ref_e:
                skipped_systems += 1
                continue

            deltas = per_step_deltas(
                np.asarray(ref_e), np.asarray(ref_g),
                np.asarray(dft_e), np.asarray(dft_g),
                n_atoms=n_atoms,
            )
            per_system.append(deltas)

            partition = fragments_per_struct.get(struct_name)
            if partition is not None and struct_name not in bad_partition:
                try:
                    per_system_parts.append(decompose_force_errors_rigid(
                        deltas["delta_force_meV_per_A"], np.asarray(pos),
                        masses, partition,
                    ))
                except ValueError as exc:
                    bad_partition.add(struct_name)
                    logger.warning(
                        f"  {struct_name}: fragment partition unusable "
                        f"({exc}) — excluded from the force decomposition."
                    )

        if not per_system:
            logger.warning(
                f"  {functional}: no complete trajectory data — excluded."
            )
            continue

        metrics[functional] = summarize_method_metrics(per_system)
        raw_deltas[functional] = {
            "delta_e_per_atom_meV": np.concatenate(
                [d["delta_e_per_atom_meV"] for d in per_system]
            ),
            "delta_force_meV_per_A": np.concatenate(
                [d["delta_force_meV_per_A"].ravel() for d in per_system]
            ),
        }
        if per_system_parts:
            metrics[functional].update(summarize_force_decomposition(per_system_parts))
            for part in FORCE_PART_KEYS:
                raw_deltas[functional][f"delta_{part}"] = np.concatenate(
                    [p[part] for p in per_system_parts]
                )
        if skipped_systems:
            logger.info(
                f"  {functional}: {skipped_systems} systems had no complete "
                "records."
            )
    return metrics, raw_deltas


def build_ranking_df(
    rmsd_df: pd.DataFrame, traj_metrics: dict,
    dft_geom_functionals: dict, score_weights: dict,
    force_key: str = "rmsd_force",
):
    """Merge eq-RMSD and trajectory force-RMSD keyed by functional and
    compute the z-score combined ranking.

    ``rmsd_df`` columns are mixed-case (built by ``compare_all_rmsd``);
    ``traj_metrics`` keys are lowercase (the workflow convention). We
    look up in mixed case but emit lowercase keys so the intersection
    with ``traj_metrics`` is well-defined.

    ``force_key`` names the entry of ``traj_metrics`` that stands in for
    ``score_weights['rmsd_force']`` (``rmsd_force`` or
    ``rmsd_force_intermolecular``); the ranking column carries that
    name so the methodology block labels it correctly. Functionals
    lacking it are left out.

    Energy is not part of the geom_benchmark ranking — absolute energies
    aren't meaningful for geometry quality. ``rmsd_energy`` stays
    available in ``traj_metrics`` as a side-channel diagnostic.
    """
    rmsd_eq = {}
    for group, funcs in dft_geom_functionals.items():
        for f in funcs:
            col = f"{group}_{f}"
            if col in rmsd_df.columns:
                mean_val = float(rmsd_df[col].dropna().mean())
                if not np.isnan(mean_val):
                    rmsd_eq[f.lower()] = mean_val

    common = sorted(
        m for m in set(rmsd_eq.keys()) & set(traj_metrics.keys())
        if force_key in traj_metrics[m]
        and not np.isnan(traj_metrics[m][force_key])
    )
    if not common:
        return None

    metrics_df = pd.DataFrame(
        [
            {
                "rmsd_eq": rmsd_eq[m],
                force_key: traj_metrics[m][force_key],
            }
            for m in common
        ],
        index=common,
    )
    weights = {
        (force_key if k == "rmsd_force" else k): v
        for k, v in score_weights.items()
    }
    ranking_df = combined_zscore_ranking(metrics_df, weights)
    return ranking_df.join(metrics_df, how="left")


def run_trajectory_analysis(
    *, config, client, odset_dict: dict, geom_ref_opt_lot: str,
    all_dft_functionals: List[str], dft_geom_functionals: dict,
    dft_program: str, dft_keyword, dft_tag: str,
    rmsd_df: pd.DataFrame, res_folder: Path, logger: logging.Logger,
    fragments_per_struct: dict = None,
):
    """Top-level driver for the trajectory benchmark.

    Pulls each system's reference trajectory, builds a per-system
    ``SinglepointDataset``, registers one SP+gradient spec per DFT
    functional, submits, waits for completion, computes per-method
    MAE/RMSE of energy and forces vs the reference, builds the combined
    z-score-weighted ranking, logs the BENCHMARK-RESULTS-style per-group
    tables + the ranking table, and writes the JSON + plot artifacts
    next to the existing geom_benchmark outputs.

    Returns ``(ranking_df, raw_deltas)``.
    """
    # Convention used by be_hess / energy_benchmark / sampling: lowercase
    # spec names once at the workflow entry, then pass through untouched.
    # qcf-astrochem stores specs lowercase and lookups are case-sensitive.
    # dft_geom_functionals stays mixed-case here because compare_all_rmsd
    # built rmsd_df with mixed-case column names; build_ranking_df does
    # the lookup in mixed case and emits lowercase keys.
    all_dft_functionals = [s.lower() for s in all_dft_functionals]
    padded_log(logger, "Trajectory analysis (DFT vs reference)")

    logger.info("\nCollecting reference trajectories…")
    traj_data = collect_reference_trajectories(
        odset_dict, geom_ref_opt_lot, logger,
        dedup_rmsd=getattr(config, "trajectory_dedup_rmsd", None),
        dedup_min_frames=getattr(config, "trajectory_dedup_min_frames", 1),
    )
    if not traj_data:
        logger.warning(
            "\n  No usable reference trajectories — skipping trajectory "
            "analysis.\n"
        )
        return None, {}

    logger.info("\nBuilding trajectory singlepoint datasets…")
    traj_dsets = build_trajectory_sp_datasets(
        client, traj_data, all_dft_functionals,
        dft_program, dft_keyword, logger,
    )

    tag_nl = getattr(config, "tag_nl_geometry", None) or dft_tag
    logger.info(
        f"\nSubmitting trajectory SP+gradient computations to tag "
        f"'{dft_tag}' (VV10 '-V' -> '{tag_nl}')…"
    )
    submit_trajectory_singlepoints(
        traj_dsets, all_dft_functionals, dft_tag, logger, tag_nl=tag_nl,
    )

    logger.info("\nWaiting for trajectory SPs to complete…")
    wait_for_trajectory_completion(
        traj_dsets, all_dft_functionals, wait_interval=200, logger=logger,
    )

    logger.info("\nComputing per-method trajectory metrics…")
    traj_metrics, raw_deltas = compute_per_method_trajectory_metrics(
        traj_data, traj_dsets, all_dft_functionals, logger,
        fragments_per_struct=fragments_per_struct,
    )

    ranking_metric = getattr(config, "ranking_metric", "total")
    decomposed = any("rmsd_force_intermolecular" in m for m in traj_metrics.values())
    if ranking_metric == "intermolecular" and not decomposed:
        logger.warning(
            "\n  ranking_metric = 'intermolecular' needs 'fragments' in the "
            "config; no fragment decomposition available — ranking by the "
            "total force RMSD instead.\n"
        )
        ranking_metric = "total"
    force_key = RANKING_FORCE_KEYS[ranking_metric]
    score_weights = {
        (force_key if k == "rmsd_force" else k): v
        for k, v in config.score_weights.items()
    }

    ranking_df = build_ranking_df(
        rmsd_df, traj_metrics, dft_geom_functionals, config.score_weights,
        force_key=force_key,
    )

    padded_log(logger, "TRAJECTORY BENCHMARK RESULTS")
    log_trajectory_metrics_per_group(
        logger, traj_metrics, dft_geom_functionals, ranking_df,
    )

    if decomposed:
        padded_log(logger, "TRAJECTORY FORCE-ERROR DECOMPOSITION (meV/Å; torque meV)")
        log_nm_force_decomposition(
            logger, traj_metrics, dft_geom_functionals,
            sort_key=force_key,
            sort_label={"total": "Total", "intermolecular": "Net force"}[ranking_metric],
            cols=TRAJECTORY_DECOMPOSITION_COLS,
            present_key="rmsd_force_intermolecular",
        )
        logger.info("")
        logger.info("  Net force / Torque = error of the force / torque on each fragment")
        logger.info("              as a whole; forces between atoms of the same fragment")
        logger.info("              cancel, so both are purely intermolecular.")
        logger.info("  Intramol. = remainder after projecting out each fragment's rigid")
        logger.info("              translation and rotation: zero net force and torque,")
        logger.info("              i.e. bond-length and angle errors (stiff stretches")
        logger.info("              dominate the Total).")
        logger.info(f"  Rows sorted by the ranking metric (ranking_metric = '{ranking_metric}');")
        logger.info("  the same metric enters the combined ranking below as RMSD_F.")

    padded_log(logger, "Combined ranking (z-score weighted)")
    log_trajectory_ranking_table(
        logger, ranking_df, score_weights=score_weights,
    )

    folder_path_json = res_folder / "json"
    folder_path_json.mkdir(parents=True, exist_ok=True)
    metrics_json = {
        m: {k: v for k, v in d.items()}
        for m, d in traj_metrics.items()
    }
    (folder_path_json / "results_trajectory_metrics.json").write_text(
        json.dumps(metrics_json, indent=4, default=str)
    )
    if ranking_df is not None:
        ranking_df.to_json(
            str(folder_path_json / "results_trajectory_ranking.json"),
        )

    # Persist raw per-functional force/energy delta arrays for downstream
    # histogram / distribution analysis. Each functional contributes two
    # flat arrays — `<functional>__force` (meV/Å, per Cartesian component)
    # and `<functional>__energy` (meV/atom). Load with
    # ``data = np.load(path)``; iterate with ``data.files``.
    if raw_deltas:
        npz_payload = {}
        for functional, d in raw_deltas.items():
            npz_payload[f"{functional}__force"] = np.asarray(
                d["delta_force_meV_per_A"]
            )
            npz_payload[f"{functional}__energy"] = np.asarray(
                d["delta_e_per_atom_meV"]
            )
            for part in FORCE_PART_KEYS:
                if f"delta_{part}" in d:
                    npz_payload[f"{functional}__force_{part}"] = np.asarray(
                        d[f"delta_{part}"]
                    )
        np.savez_compressed(
            res_folder / "raw_deltas_trajectory.npz", **npz_payload,
        )

    logger.info(
        f"\n  Trajectory metrics + ranking + raw deltas written to "
        f"{res_folder}/\n"
    )

    if raw_deltas and getattr(config, "generate_plots", False):
        folder_path_plots = res_folder / "plots"
        folder_path_plots.mkdir(parents=True, exist_ok=True)
        try:
            trajectory_error_histograms(
                raw_deltas, mol_name=config.opt_dataset,
                plot_path=str(folder_path_plots),
                ranking_df=ranking_df,
            )
            logger.info(
                f"  Trajectory error histograms written to "
                f"{folder_path_plots}/\n"
            )
        except Exception as exc:
            logger.warning(
                f"  Histogram plotting failed: {exc} — "
                f"metrics are still available in json/"
            )

    return ranking_df, raw_deltas
