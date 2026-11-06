"""Geometry benchmark workflow — entry-based, single OptimizationDataset."""
import time
import logging
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path
from qcelemental.models.molecule import Molecule

from ..models.geom_benchmark import GeomBenchmarkConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import (
    padded_log, log_dataframe_averages, log_progress, dict_to_log, beep_banner,
)
from ..core.dft_functionals import (
    geom_hmgga_dz, geom_hmgga_tz, geom_gga_dz, geom_gga_tz, geom_sqm_mb,
)
from ..core.plotting_utils import rmsd_histograms
from ..core.benchmark_utils import compute_rmsd
from ..core.trajectory_workflow import run_trajectory_analysis
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient, is_complete, is_incomplete, is_error, status_label

bcheck = "\u2714"
mia0911 = "\u2606"
gear = "\u2699"

welcome_msg = beep_banner(
    "Geometry Benchmark",
    tagline="Shine, Loom, Manifest.",
    authors="Stefan Vogt-Geisse",
)


def create_and_add_specification(client, odset, method, basis, program,
                                  qc_keyword, geom_keywords=None):
    logger = logging.getLogger("beep")
    spec_name = f"{method}_{basis}".lower()

    spec = {
        "name": spec_name,
        "description": f"Geometric {program}/{method}/{basis}",
        "optimization_spec": {"program": "geometric", "keywords": geom_keywords},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            # Inline keyword dict (validated by GeomBenchmarkConfig); the
            # adapter raises on anything that is not a dict or None.
            "keywords": dict(qc_keyword or {}),
            "program": program,
        },
    }
    qcf.add_opt_specification(odset, spec, overwrite=True)
    logger.debug(f"Created and added the specification {spec_name} to {odset.name}")
    return spec_name


def optimize_reference_molecule(odset, struct_name, geom_ref_opt_lot, opt_tag):
    """Submit the reference-LOT optimization for one entry.

    Charge/multiplicity come from the entry's stored molecule — no global
    adsorbate multiplicity is assumed.
    """
    return qcf.submit_optimizations(odset, geom_ref_opt_lot, tag=opt_tag, subset={struct_name})


def optimize_dft_molecule(client, odset, struct_name, method, basis, program,
                           dft_keyword, opt_tag):
    spec_name = create_and_add_specification(client, odset, method, basis, program, dft_keyword)
    return qcf.submit_optimizations(odset, spec_name, tag=opt_tag, subset={struct_name})


def wait_for_completion(client, odset_dict, opt_lot, program,
                         wait_interval=600, check_errors=False,
                         ref_spec=None):
    logger = logging.getLogger("beep")
    if isinstance(opt_lot, str):
        opt_lot = [opt_lot]

    ref_spec_key = ref_spec.lower() if ref_spec else None

    logger.info("\nChecking if the computations have finished\n")
    while True:
        dft_complete = 0
        dft_incomplete = 0
        dft_error = 0
        ref_complete = 0
        ref_incomplete = 0
        ref_error = 0

        for lot in opt_lot:
            lot_key = lot.lower()
            for struct_name, odset in odset_dict.items():
                record = odset.get_record(struct_name, lot_key)
                if record is None:
                    continue
                if is_error(record.status) and check_errors:
                    raise RuntimeError(
                        f"Error encountered in computation for {struct_name} with spec '{lot_key}'"
                    )
                if is_complete(record.status):
                    dft_complete += 1
                elif is_incomplete(record.status):
                    dft_incomplete += 1
                elif is_error(record.status):
                    dft_error += 1

        if ref_spec_key:
            for struct_name, odset in odset_dict.items():
                record = odset.get_record(struct_name, ref_spec_key)
                if record is None:
                    continue
                if is_complete(record.status):
                    ref_complete += 1
                elif is_incomplete(record.status):
                    ref_incomplete += 1
                elif is_error(record.status):
                    ref_error += 1

        total_incomplete = dft_incomplete + ref_incomplete

        if total_incomplete == 0:
            if ref_spec:
                logger.info(
                    f"Reference [{ref_spec}]: Complete: {ref_complete}, "
                    f"Error: {ref_error}"
                )
            logger.info(
                f"DFT: Complete: {dft_complete}, Error: {dft_error}"
            )
            logger.info(f"\nAll entries have been processed. {bcheck}")
            return dft_complete + ref_complete

        status_parts = []
        if ref_spec:
            status_parts.append(
                f"Ref: {ref_complete} done, {ref_incomplete} running, {ref_error} err"
            )
        status_parts.append(
            f"DFT: {dft_complete} done, {dft_incomplete} running, {dft_error} err"
        )
        logger.info(
            f"  {' | '.join(status_parts)}"
        )
        logger.info(
            f"  Waiting {wait_interval}s before rechecking..."
        )
        time.sleep(wait_interval)


def compare_rmsd(dft_lot, odset_dict, ref_geom_fmols):
    logger = logging.getLogger("beep")
    logger.propagate = False
    rmsd_df = pd.DataFrame(index=odset_dict.keys(), columns=dft_lot)
    final_opt_lot = {}
    total_operations = len(dft_lot)

    errored_specs = []
    for i, opt_lot in enumerate(dft_lot):
        opt_lot_key = opt_lot.lower()
        rmsd_tot_dict = {}
        err = None
        for struct_name, odset in odset_dict.items():
            record = odset.get_record(struct_name, opt_lot_key)
            if record is None:
                # qcportal 0.64 get_record returns None for a missing record.
                err = True
                logger.warning(
                    f"WARNING: No record for {struct_name} at the {opt_lot} level "
                    "of theory (never submitted or deleted). "
                    "This level of theory will be excluded from the benchmark."
                )
                errored_specs.append((opt_lot, struct_name, None))
                break
            err = is_error(record.status) or record.status.value == "cancelled"
            if err:
                logger.warning(
                    f"WARNING: Calculation for {struct_name} at the {opt_lot} level of theory "
                    f"finished with error (record id: {record.id}). "
                    f"This level of theory will be excluded from the benchmark."
                )
                errored_specs.append((opt_lot, struct_name, record.id))
                break
            fmol = record.final_molecule
            rmsd = compute_rmsd(ref_geom_fmols[struct_name], fmol, rmsd_symm=True)
            rmsd_tot_dict[struct_name] = rmsd
            rmsd_df.at[struct_name, opt_lot] = rmsd

        if err:
            rmsd_df[opt_lot] = np.nan
            continue
        rmsd_tot = list(rmsd_tot_dict.values())
        final_opt_lot[opt_lot] = np.mean(rmsd_tot)
        log_progress(logger, i + 1, total_operations)

    if errored_specs:
        logger.warning(f"\nSummary of errored optimizations ({len(errored_specs)} total):")
        for spec, struct, rec_id in errored_specs:
            rec_label = f"record id: {rec_id}" if rec_id is not None else "no record"
            logger.warning(f"  {spec} / {struct}  ({rec_label})")
        logger.warning("")

    rmsd_df = rmsd_df.dropna(axis=1, how="all")
    lowest_values = sorted(final_opt_lot.values())[:1]
    best_geom_lot = {k: v for k, v in final_opt_lot.items() if v in lowest_values}
    return best_geom_lot, final_opt_lot, rmsd_df


def compare_all_rmsd(functional_groups, odset_dict, ref_geom_fmols):
    logger = logging.getLogger("beep")
    best_opt_lot = {}
    combined_rmsd_df = pd.DataFrame()

    for func_group, functionals in functional_groups.items():
        logger.info(f"\nProcessing RMSD for {func_group} type methods:")
        group_best_opt_lot, final_opt_lot, rmsd_df = compare_rmsd(
            functionals, odset_dict, ref_geom_fmols
        )
        rmsd_df.columns = [f"{func_group}_{col}" for col in rmsd_df.columns]
        combined_rmsd_df = pd.concat([combined_rmsd_df, rmsd_df], axis=1)
        best_opt_lot[func_group] = group_best_opt_lot

    return best_opt_lot, combined_rmsd_df


def run(config: GeomBenchmarkConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    opt_dset_name = config.opt_dataset

    # Create output folder: <cwd>/<opt_dataset>/
    res_folder = Path.cwd() / opt_dset_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"geom_benchmark_{opt_dset_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"geom_benchmark_{opt_dset_name}.json"
    config_path.write_text(safe_config_dump(config))

    logger.info(welcome_msg)

    hl_tag = config.tag_reference_geometry
    dft_tag = config.tag_dft_geometry
    gr_method, gr_basis, gr_program = config.reference_geometry_level_of_theory
    geom_ref_opt_lot = (gr_method + "_" + gr_basis).lower()

    bchmk_structs = config.benchmark_structures

    padded_log(logger, "Starting BEEP geometry benchmark procedure", padding_char=gear)
    logger.info(f"Optimization dataset: {opt_dset_name}")
    logger.info(f"Benchmark Structures: {bchmk_structs}")
    fragments_per_struct = None
    if config.fragments is not None:
        fragments_per_struct = {
            name: [list(f) for f in partition]
            for name, partition in config.fragments.items()
        }
        logger.info("Fragments for the trajectory force decomposition (per structure):")
        for name in bchmk_structs:
            p = fragments_per_struct[name]
            n_atoms = sum(len(f) for f in p)
            logger.info(f"  {name}: {len(p)} fragments ({n_atoms} atoms) → {p}")
        logger.info(f"Trajectory ranking metric: {config.ranking_metric}")

    qcf.check_collection_existence(client, opt_dset_name)
    opt_dset = qcf.get_collection(client, "OptimizationDataset", opt_dset_name)

    # Every benchmark entry lives in the same dataset. Each stored molecule
    # carries its own charge/multiplicity. To benchmark a monomer or bare
    # surface alongside the complexes, add it as another entry in opt_dataset.
    missing = [n for n in bchmk_structs if n not in set(opt_dset.entry_names)]
    if missing:
        raise KeyError(
            f"benchmark_structures not found in {opt_dset_name}: {missing}"
        )
    odset_dict = {name: opt_dset for name in bchmk_structs}

    padded_log(logger, "Start of the geometry refrence processing")
    logger.info(f"Method: {gr_method}")
    logger.info(f"Basis: {gr_basis}")
    logger.info(f"Program: {gr_program}\n")
    gr_keywords = config.reference_geometry_keywords
    create_and_add_specification(
        client, opt_dset, method=gr_method, basis=gr_basis,
        program=gr_program, qc_keyword=gr_keywords, geom_keywords=None,
    )

    ct = 0
    for struct_name, odset in odset_dict.items():
        meta = optimize_reference_molecule(odset, struct_name, geom_ref_opt_lot, hl_tag)
        ct += getattr(meta, 'n_inserted', 0) + getattr(meta, 'n_existing', 0)

    logger.info(
        f"\nSend a total of {ct} structures to compute at the "
        f"{geom_ref_opt_lot} level of theory to the tag {hl_tag}\n"
    )

    padded_log(logger, "Start of the DFT geometry computations")

    dft_program = config.dft_optimization_program
    dft_keyword = config.qc_keywords

    dft_geom_functionals = {
        "geom_hmgga_dz": geom_hmgga_dz(),
        "geom_hmgga_tz": geom_hmgga_tz(),
        "geom_gga_dz": geom_gga_dz(),
        "geom_gga_tz": geom_gga_tz(),
        "geom_sqm_mb": geom_sqm_mb(),
    }

    all_dft_functionals = [
        functional
        for functionals in dft_geom_functionals.values()
        for functional in functionals
    ]

    logger.info(f"Program: {dft_program}")
    logger.info(f"DFT and SQM geometry methods:")
    dict_to_log(logger, dft_geom_functionals)

    ct = 0
    c = 0
    padded_log(logger, "Start sending DFT optimizations")
    for struct_name, odset in odset_dict.items():
        logger.info(f"\nSending geometry optimizations for {struct_name}")
        cs = 0
        for functionals in dft_geom_functionals.values():
            for functional in functionals:
                method, basis = functional.split("_", 1)
                meta = optimize_dft_molecule(
                    client, odset, struct_name, method, basis,
                    dft_program, dft_keyword, dft_tag,
                )
                n = getattr(meta, 'n_inserted', 0) + getattr(meta, 'n_existing', 0)
                cs += n
                ct += n
                c += 1
        logger.info(f"Send {cs} geometry optimizations for structure {struct_name}")

    logger.info(f"\nSend {ct}/{c} to the tag {dft_tag}\n")

    wait_for_completion(
        client, odset_dict, all_dft_functionals, dft_program,
        wait_interval=200, check_errors=False,
        ref_spec=geom_ref_opt_lot,
    )

    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, geom_ref_opt_lot)
        if config.use_initial_reference_geometry:
            ref_geom_fmols[struct_name] = record.initial_molecule
        else:
            ref_geom_fmols[struct_name] = record.final_molecule

    padded_log(
        logger,
        "Start of RMSD comparison between DFT and {} geometries",
        geom_ref_opt_lot,
    )

    best_opt_lot, rmsd_df = compare_all_rmsd(dft_geom_functionals, odset_dict, ref_geom_fmols)

    padded_log(logger, "BENCHMARK RESULTS")
    log_dataframe_averages(logger, rmsd_df)

    folder_path_json = data_folder / "json"
    folder_path_json.mkdir(parents=True, exist_ok=True)

    rmsd_df.to_json(str(folder_path_json / "results_geom_benchmark.json"))
    logger.info(f"\nDataFrame successfully saved to {folder_path_json}/results_geom_benchmark.json\n")

    if config.generate_plots:
        folder_path_plots = data_folder / "plots"
        folder_path_plots.mkdir(parents=True, exist_ok=True)
        rmsd_histograms(rmsd_df, opt_dset_name, str(folder_path_plots))

    # Trajectory analysis: SP+gradient at every reference-trajectory geometry,
    # MAE/RMSE of E and forces vs reference, combined z-score ranking.
    # Runs AFTER BENCHMARK RESULTS so the eq-RMSD per-group output is the
    # first benchmark summary the user sees; the trajectory benchmark
    # appears immediately below in the same per-group style.
    if config.trajectory_analysis:
        run_trajectory_analysis(
            config=config, client=client, odset_dict=odset_dict,
            geom_ref_opt_lot=geom_ref_opt_lot,
            all_dft_functionals=all_dft_functionals,
            dft_geom_functionals=dft_geom_functionals,
            dft_program=dft_program, dft_keyword=dft_keyword,
            dft_tag=dft_tag, rmsd_df=rmsd_df,
            res_folder=data_folder, logger=logger,
            fragments_per_struct=fragments_per_struct,
        )

    padded_log(
        logger,
        "Geometry Benchmark finished successfully! Hasta pronto!",
        padding_char=mia0911,
    )

    logger.removeHandler(file_handler)
    file_handler.close()
