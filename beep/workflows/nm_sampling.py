"""Normal-mode displacement benchmark workflow.

Entry point for ``beep --config <nm_sampling.json>``. Sets up logging
and the output folder, resolves the optimisation datasets for the
target molecule + surface model + benchmark binding sites, and hands
control to the orchestrator in
:mod:`beep.core.nm_sampling_workflow`.

The actual physics + orchestration lives in:
  - :mod:`beep.core.normal_mode_sampling` — classification, selection,
    displacement math.
  - :mod:`beep.core.nm_sampling_workflow` — chain Hessian → vibanal →
    displacement → SP+gradient → metrics.
  - :mod:`beep.adapters.qcfractal_adapter` — QCFractal I/O (``submit_hessians``,
    ``fetch_normal_modes``, the ``SinglepointDataset`` helpers).
"""
import logging
from pathlib import Path

from ..models.nm_sampling import NmSamplingConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import padded_log, beep_banner
from ..core.dft_functionals import (
    geom_hmgga_dz, geom_hmgga_tz, geom_gga_dz, geom_gga_tz, geom_sqm_mb,
    geom_nl,
)
from ..core.nm_sampling_workflow import run_nm_sampling
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient


mia0911 = "☆"
gear = "⚙"

welcome_msg = beep_banner(
    "NM-Sampling",
    tagline="Move along the soft modes; let CCSD(T) keep score.",
    authors="Stefan Vogt-Geisse",
)


def run(config: NmSamplingConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    opt_dset_name = config.opt_dataset

    res_folder = Path.cwd() / opt_dset_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    log_file = res_folder / f"nm_sampling_{opt_dset_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    config_path = res_folder / f"nm_sampling_{opt_dset_name}.json"
    config_path.write_text(safe_config_dump(config))

    logger.info(welcome_msg)

    bchmk_structs = config.benchmark_structures
    fragments = {
        name: [list(f) for f in partition]
        for name, partition in config.fragments.items()
    }

    padded_log(logger, "Starting BEEP NM-sampling benchmark", padding_char=gear)
    logger.info(f"Optimization dataset: {opt_dset_name}")
    logger.info(f"Benchmark structures: {bchmk_structs}")
    logger.info("Fragments (per structure):")
    for name in bchmk_structs:
        p = fragments[name]
        n_atoms = sum(len(f) for f in p)
        logger.info(f"  {name}: {len(p)} fragments ({n_atoms} atoms) → {p}")
    logger.info(f"Geometry LOT: {config.geometry_opt_lot}")
    logger.info(f"Hessian LOT:  {config.hessian_lot}")
    logger.info(f"Reference grad LOT: {config.reference_grad_lot}\n")

    qcf.check_collection_existence(client, opt_dset_name)
    opt_dset = qcf.get_collection(client, "OptimizationDataset", opt_dset_name)

    # Every benchmark entry pulls from the same dataset. Same {name → dataset}
    # shape as before so downstream orchestration doesn't need to change.
    odset_dict = {name: opt_dset for name in bchmk_structs}

    # Same five functional groups as geom_benchmark — the displacements
    # are evaluated at every functional in this list.
    dft_geom_functionals = {
        "geom_hmgga_dz": geom_hmgga_dz(),
        "geom_hmgga_tz": geom_hmgga_tz(),
        "geom_gga_dz":   geom_gga_dz(),
        "geom_gga_tz":   geom_gga_tz(),
        "geom_sqm_mb":   geom_sqm_mb(),
    }
    # Optional VV10 "-V" group, routed to ORCA+SCNL in build_nm_sp_datasets
    # (Psi4 has no analytic VV10 gradient).
    if config.include_nl_functionals:
        dft_geom_functionals["geom_nl"] = geom_nl()
        logger.info("include_nl_functionals=True -> adding VV10 group (ORCA+SCNL)")
    all_dft_functionals = [
        f for group in dft_geom_functionals.values() for f in group
    ]

    metrics, _raw_deltas = run_nm_sampling(
        config=config, client=client, odset_dict=odset_dict,
        all_dft_functionals=all_dft_functionals,
        dft_geom_functionals=dft_geom_functionals,
        fragments_per_struct=fragments,
        res_folder=data_folder, logger=logger,
    )

    if metrics is None:
        # run_nm_sampling returns (None, {}) when it aborts (no geometries,
        # no normal modes, no displacements, ...). Do not report success:
        # fail the CLI run with a nonzero exit code, as an uncaught exception
        # in any other workflow would.
        logger.error(
            "NM-sampling benchmark aborted before producing results; "
            f"see the log above and {log_file} for the reason."
        )
        logger.removeHandler(file_handler)
        file_handler.close()
        raise SystemExit(1)

    padded_log(
        logger,
        "NM-sampling benchmark finished successfully! Hasta pronto!",
        padding_char=mia0911,
    )
    logger.removeHandler(file_handler)
    file_handler.close()
