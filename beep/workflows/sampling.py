"""Sampling workflow — refactored from workflows/launch_sampling.py."""
import logging
from pathlib import Path

from ..models.sampling import SamplingConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import padded_log, beep_banner
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient

bcheck = "\u2714"
gear = "\u2699"

welcome_msg = beep_banner(
    "Set-of-clusters Sampling",
    quote="Adopt the pace of nature: her secret is patience.",
    quote_author="Ralph Waldo Emerson",
    tagline="Seek Locate Map",
    authors="Stefan Vogt-Geisse and Giulia M. Bovolenta",
)


def config_summary_msg(config):
    """Format a clean summary of the sampling configuration."""
    separator = "-" * 88
    s_lot = config.sampling_level_of_theory
    r_lot = config.refinement_level_of_theory
    lines = [
        "",
        separator,
        f"  Molecule:             {config.molecule}",
        f"  Surface model:        {config.surface_model_collection}",
        f"  Small molecule coll:  {config.small_molecule_collection}",
        f"  Sampling LOT:         {s_lot.display}",
        f"  Refinement LOT:       {r_lot.display}",
        f"  Sampling shell:       {config.sampling_shell} Angstrom",
        f"  Sampling condition:   {config.sampling_condition}",
        f"  RMSD cutoff:          {config.rmsd_value}",
        f"  RMSD symmetry:        {config.rmsd_symmetry}",
        f"  Target binding sites: {config.total_binding_sites}",
        separator,
        "",
    ]
    return "\n".join(lines)


def process_refinement(client, ropt_lot_name, rmethod, rbasis, program,
                       qc_keyword, ds_opt, logger, rtag="refinement",
                       lot_display=None, refinement_opt_keywords=None):
    lot_display = lot_display or f"{rmethod}/{rbasis}/{program}"
    spec = {
        "name": ropt_lot_name,
        "description": f"Geometric + {lot_display}",
        "optimization_spec": {
            "program": "geometric",
            "keywords": refinement_opt_keywords or None,
        },
        "qc_spec": {
            "driver": "gradient",
            "method": rmethod,
            "basis": rbasis,
            "keywords": qc_keyword,
            "program": program,
        },
    }

    qcf.add_opt_specification(ds_opt, spec, overwrite=True)
    c = qcf.submit_optimizations(ds_opt, ropt_lot_name, tag=rtag)

    logger.info(
        f"\nRefinement optimization initiated with specification '{ropt_lot_name}' \n"
        f"using {lot_display}. \n"
        f"Description: {spec['description']}. \n"
        f"Tag applied: '{rtag}'\n"
        f"Optimizations submitted: {c}. {bcheck} \n"
    )


def _filter_and_register(opt_molecules_new, refinement_opt_dset, target_mol,
                         rmsd_val, rmsd_symm, logger):
    """RMSD-filter completed sampling structures against the refinement
    dataset and register the unique ones there. Returns the number added.
    Shared by the top-up path (``run_sampling``) and the harvest path."""
    from ..core.sampling import filter_binding_sites

    existing = [
        (entry.name, entry.initial_molecule)
        for entry in refinement_opt_dset.iterate_entries()
    ]
    logger.info(
        f"Filtering {len(opt_molecules_new)} new molecules against existing "
        f"{len(existing)} molecules using an RMSD criteria of {rmsd_val}"
    )
    unique_mols = filter_binding_sites(
        opt_molecules_new, existing, cut_off_val=rmsd_val,
        rmsd_symm=rmsd_symm, ligand_size=len(target_mol.symbols), logger=logger,
        grid=0.5, nb_radius=4, dm_tau=1e-3,
    )
    for entry_name, mol_obj in unique_mols:
        try:
            qcf.add_opt_entry(refinement_opt_dset, entry_name, mol_obj)
        except KeyError as e:
            logger.info(f"{e} in {refinement_opt_dset.name}")
    return len(unique_mols)


def harvest_cluster(client, sampling_opt_dset, refinement_opt_dset, opt_lot,
                    target_mol, rmsd_val, rmsd_symm, logger, wait_frequency=120):
    """Collect finished sampling optimizations into the refinement dataset
    without generating any new placements.

    This is the resume path. Waits for sampling optimizations still in
    flight (including records reset since the previous run), then
    RMSD-filters every COMPLETE sampling structure not yet present in the
    refinement dataset and registers the unique ones. Structures rejected
    as duplicates in an earlier run are re-tested (cheap: the filter has a
    grid prefilter) and rejected again. Returns the number of binding
    sites added.
    """
    entries = list(sampling_opt_dset.entry_names)
    if not entries:
        logger.info("  No sampling entries yet; nothing to harvest.")
        return 0

    pids = qcf.get_job_ids(sampling_opt_dset, entries, opt_lot)
    if pids:
        qcf.wait_for_completion(client, pids, wait_frequency, logger)

    refined = set(refinement_opt_dset.entry_names)
    candidates = [e for e in entries if e not in refined]
    if not candidates:
        logger.info("  Every completed sampling structure is already in the refinement dataset.")
        return 0

    complete = qcf.fetch_opt_molecules(
        sampling_opt_dset, candidates, opt_lot, status="COMPLETE",
    )
    logger.info(
        f"  Harvest: {len(complete)} completed, not-yet-refined sampling "
        f"structure(s) out of {len(candidates)} candidate(s)."
    )
    if not complete:
        return 0
    return _filter_and_register(
        complete, refinement_opt_dset, target_mol, rmsd_val, rmsd_symm, logger,
    )


def run_sampling(
    method: str,
    basis: str,
    program: str,
    tag: str,
    kw_id,
    sampling_opt_dset,
    refinement_opt_dset,
    opt_lot: str,
    rmsd_symm: bool,
    store_initial: bool,
    rmsd_val: float,
    target_mol,
    cluster,
    debug_path,
    client,
    sampling_shell: float,
    sampling_condition: str,
    logger,
    sampling_method: str = "adaptive",
    sampling_opt_keywords=None,
):
    """
    Top-up path: generate new placements in the free entry-name slots,
    optimize them, RMSD-filter against the refinement dataset. Existing
    entries are waited on and re-filtered too, so after a harvest pass
    this only adds work for the new placements.
    """
    from ..core.molecule_sampler import random_molecule_sampler as mol_sample

    FREQUENCY = 120
    ATOMS_PER_CLUSTER_MOL = 3
    binding_site_num = 0
    n_smpl_mol = 0

    shell_list = [sampling_shell]  # adaptive: one conformal pass; the level sets density

    # Candidate count is driven by the sampling level: a fraction of the accessible
    # surface anchors times orientations per anchor (see SAMPLING_LEVELS). This
    # replaces the old n_water//3 heuristic, which badly undersampled large clusters
    # (W200: 66 vs ~371 real anchors), and the multi-shell height passes (now
    # redundant -- adsorption height is sampled per anchor in one pass).
    from ..core.molecule_sampler import adaptive_shift_vectors, SAMPLING_LEVELS
    frac, n_orient, _jit = SAMPLING_LEVELS.get(
        sampling_condition, SAMPLING_LEVELS["normal"]
    )
    try:
        n_anchors = len(adaptive_shift_vectors(cluster, target_mol, sampling_shell))
        max_structures = max(3, int(-(-(frac * n_anchors) // 1)) * n_orient)  # ceil * orient
    except Exception as exc:  # degenerate/mocked geometry -> fall back to the size heuristic
        max_structures = int(max(3, (len(cluster.symbols) / ATOMS_PER_CLUSTER_MOL) // 3))
        logger.warning(
            "Adaptive anchor count failed (%s: %s); falling back to the "
            "n_water//3 size heuristic: %d structures.",
            type(exc).__name__, exc, max_structures,
        )

    logger.info(
        f"Entering the sampling procedure, will generate a total of "
        f"{max_structures} structures for each shell. "
        f"{len(shell_list)} shells will be sampled."
    )

    if program == "terachem":
        if len(method.split("-")) == 2:
            method = method.split("-")[0]

    # Build specification dict for the adapter
    _smpl_opt_kw = {"maxiter": 125, **(sampling_opt_keywords or {})}
    spec = {
        "name": opt_lot,
        "description": "Geometric Optimization",
        "optimization_spec": {"program": "geometric", "keywords": _smpl_opt_kw},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": kw_id if isinstance(kw_id, dict) else {},
            "program": program,
        },
    }
    qcf.add_opt_specification(sampling_opt_dset, spec, overwrite=False)
    logger.info(
        "Adding the specification {} to the {} OptimizationDataset.".format(
            opt_lot, sampling_opt_dset.name
        )
    )

    for shell in shell_list:
        logger.info(f"\nStarting sampling within a shell of {shell} Angstrom")

        entry_name_list = []
        entry_base_name = refinement_opt_dset.name
        for i in range(max_structures):
            n_smpl_mol += 1
            entry_name = entry_base_name + "_" + str(n_smpl_mol).zfill(4)
            entry_name_list.append(entry_name)

        total_existing_entries = list(sampling_opt_dset.entry_names)

        shell_new_entries = [
            e for e in entry_name_list if e not in total_existing_entries
        ]
        shell_old_entries = [
            e for e in entry_name_list if e in total_existing_entries
        ]

        logger.info(
            f"Candidate entry name slots: {len(entry_name_list)} "
            f"({len(shell_old_entries)} already in dataset, "
            f"{len(shell_new_entries)} available)"
        )
        if shell_old_entries:
            logger.debug(
                "Existing entries: " + " ".join(shell_old_entries)
            )

        n_smpl_mol -= len(shell_new_entries)

        pid_list = qcf.get_job_ids(sampling_opt_dset, shell_old_entries, opt_lot)

        if pid_list:
            logger.debug(
                f"Procedure IDs of the optimization are: {pid_list}"
            )

        qcf.wait_for_completion(client, pid_list, FREQUENCY, logger)

        added_names = []
        if not shell_new_entries:
            logger.info(
                "All entries for this shell already exist."
            )
        else:
            max_structures = len(shell_new_entries)

            molecules, debug_mol = mol_sample(
                cluster,
                target_mol,
                sampling_shell=shell,
                max_structures=max_structures,
                debug=True,
                method=sampling_method,
                condition=sampling_condition,
            )

            logger.info(
                f"Sampler produced {len(molecules)} valid placements → "
                f"adding {len(molecules)} entries to {sampling_opt_dset.name}"
            )
            for i, m in enumerate(molecules):
                n_smpl_mol += 1
                name = shell_new_entries[i]
                try:
                    qcf.add_opt_entry(sampling_opt_dset, name, m)
                    added_names.append(name)
                except KeyError as e:
                    logger.info(e)

            if store_initial:
                logger.info(
                    f"Initial structure set for visualization will be saved "
                    f"in {str(debug_path)}"
                )
                # Strip the dot only from the filename component: applying it to
                # the whole path mangled any directory containing a dot
                # (e.g. ~/beep-0.12/, .local/).
                debug_path = Path(debug_path)
                filename = str(
                    debug_path.parent
                    / (f"{debug_path.name}_{round(shell, 2):.2f}".replace(".", "") + ".mol")
                )
                debug_mol.to_file(filename, "xyz")

        # Always submit at the current sampling LOT for the union of newly-
        # added entries plus entries already in the dataset (from a prior
        # run, possibly at a different LOT). QCFractal dedups server-side:
        # entries already computed at this spec become n_existing no-ops;
        # entries that have never been computed at this spec get fresh opts.
        present_entries = shell_old_entries + added_names
        if present_entries:
            comp_rec = qcf.submit_optimizations(
                sampling_opt_dset, opt_lot, tag=tag, subset=present_entries,
            )
            logger.info(
                f"{comp_rec.n_inserted} new, {comp_rec.n_existing} existing "
                "sampling optimization procedures."
            )

        pid_list = qcf.get_job_ids(sampling_opt_dset, present_entries, opt_lot)
        if pid_list:
            logger.info(
                "Procedure IDs of the optimization are: {}".format(
                    " ".join(str(p) for p in pid_list)
                )
            )

        qcf.wait_for_completion(client, pid_list, FREQUENCY, logger)

        opt_molecules_new = qcf.fetch_opt_molecules(
            sampling_opt_dset, present_entries, opt_lot, status="COMPLETE"
        )
        opt_mol_num = len(opt_molecules_new)
        logger.debug(
            f"{opt_mol_num} COMPLETED molecules in "
            f"{sampling_opt_dset.name} for this round, "
            f"{len(present_entries) - opt_mol_num} molecules ended in ERROR."
        )

        new_mols_count = _filter_and_register(
            opt_molecules_new, refinement_opt_dset, target_mol,
            rmsd_val, rmsd_symm, logger,
        )
        binding_site_num += new_mols_count

        logger.info(
            f"A total of {new_mols_count} unique binding sites were found "
            f"after filtering for shell {shell} Angstrom. \n"
            f"Adding this new binding sites to {refinement_opt_dset.name} "
            "for refined optimization."
        )

    total_bind_sites = len(refinement_opt_dset.entry_names)
    logger.info(
        f"Finished sampling the cluster. Found {binding_site_num} unique "
        f"binding sites. Total binding sites: {total_bind_sites}"
    )
    return None


def run(config: SamplingConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule

    # Create output folder: <cwd>/<molecule>/
    res_folder = Path.cwd() / smol_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"sampling_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"sampling_{smol_name}.json"
    config_path.write_text(safe_config_dump(config))

    logger.info(welcome_msg)

    s_lot = config.sampling_level_of_theory
    r_lot = config.refinement_level_of_theory
    method = s_lot.qc_method
    basis = s_lot.qc_basis
    program = s_lot.qc_program
    rmethod = r_lot.qc_method
    rbasis = r_lot.qc_basis
    rprogram = r_lot.qc_program

    # QC-program keywords for the refinement optimizations (dict); passed
    # verbatim as the qc_spec keywords of the refinement specification.
    qc_keyword = config.qc_keywords

    opt_lot = s_lot.lot_name
    ropt_lot = r_lot.lot_name

    # --- Configuration summary ---
    logger.info(config_summary_msg(config))

    args_dict = {
        "method": method,
        "basis": basis,
        "program": program,
        "tag": config.sampling_tag,
        "kw_id": None,  # qc_keywords is used for refinement spec, not sampling
        "rmsd_symm": config.rmsd_symmetry,
        "store_initial": config.store_initial_structures,
        "rmsd_val": config.rmsd_value,
        "sampling_shell": config.sampling_shell,
        "sampling_condition": config.sampling_condition,
        "sampling_method": config.sampling_method,
        "opt_lot": opt_lot,
        "logger": logger,
        "sampling_opt_keywords": config.sampling_opt_keywords,
    }

    # --- Validate collections ---
    logger.info("Validating collections and optimized geometries...")
    qcf.check_collection_existence(
        client, config.surface_model_collection, config.small_molecule_collection
    )
    ds_sm = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    ds_wc = qcf.get_collection(client, "OptimizationDataset", config.surface_model_collection)

    all_cluster_names = list(ds_wc.entry_names)
    if config.sampling_clusters:
        cluster_names = [c for c in all_cluster_names if c in config.sampling_clusters]
        logger.info(
            f"  Surface model clusters: {len(all_cluster_names)} available, "
            f"{len(cluster_names)} selected  ({', '.join(cluster_names)})"
        )
    else:
        cluster_names = all_cluster_names
        logger.info(f"  Surface model clusters: {len(cluster_names)}  ({', '.join(cluster_names)})")

    # Check if all the molecules are optimized at the requested level of theory.
    # Atoms (single-atom species) live in a SinglepointDataset, not an
    # OptimizationDataset, so we fetch them from the atoms_collection.
    try:
        init_mol = qcf.fetch_initial_molecule(ds_sm, smol_name, opt_lot)
        is_atom = len(init_mol.symbols) == 1
    except KeyError:
        # Molecule not in the OptimizationDataset — try the atoms collection
        init_mol = qcf.fetch_atom_molecule(client, config.atoms_collection, smol_name)
        is_atom = True

    if is_atom:
        qcf.check_optimized_molecule(ds_wc, opt_lot, cluster_names)
        args_dict["target_mol"] = init_mol
    else:
        qcf.check_optimized_molecule(ds_sm, opt_lot, [smol_name])
        qcf.check_optimized_molecule(ds_wc, opt_lot, cluster_names)
        args_dict["target_mol"] = qcf.fetch_final_molecule(ds_sm, smol_name, opt_lot)

    logger.info(f"  All geometries validated. {bcheck}\n")

    args_dict["client"] = client

    target = config.total_binding_sites

    def _submit_refinement(ds_ref):
        process_refinement(
            client, ropt_lot, rmethod, rbasis, rprogram,
            qc_keyword, ds_ref, logger, config.refinement_tag,
            lot_display=r_lot.display,
            refinement_opt_keywords=config.refinement_opt_keywords,
        )

    # --- Phase 0: pre-scan (resume table) ---
    # Open every cluster's datasets and count what previous runs left behind
    # BEFORE doing any work. A relaunch used to walk cluster by cluster,
    # topping each one up with new placements and waiting a full gfn2
    # round, only to discover at the end that the target was already met.
    datasets = {}        # w -> (ds_smplg, ds_ref)
    refined = {}         # w -> refinement entry count
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  RESUME SCAN FOR {smol_name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"  {'Cluster':<15} {'Sampled':>9} {'Complete':>9} {'Refined':>9}")
    logger.info(f"  {'-'*15} {'-'*9} {'-'*9} {'-'*9}")
    for w in cluster_names:
        ref_name = smol_name + "_" + w
        ds_smplg = qcf.get_or_create_opt_dataset(client, "pre_" + ref_name)
        ds_ref = qcf.get_or_create_opt_dataset(client, ref_name)
        datasets[w] = (ds_smplg, ds_ref)
        sampled = list(ds_smplg.entry_names)
        n_complete = 0
        if sampled:
            pids = qcf.get_job_ids(ds_smplg, sampled, opt_lot)
            if pids:
                _, counts = qcf.check_for_completion(client, pids)
                n_complete = counts.get("COMPLETE", 0)
        refined[w] = len(ds_ref.entry_names)
        logger.info(f"  {w:<15} {len(sampled):>9} {n_complete:>9} {refined[w]:>9}")
    logger.info(f"  {'-'*15} {'-'*9} {'-'*9} {'-'*9}")
    logger.info(f"  {'TOTAL':<15} {'':>9} {'':>9} {sum(refined.values()):>9}")
    logger.info(f"  Target: {target}")
    logger.info(f"{'=' * 80}\n")

    # --- Phase 1: harvest existing sampling results, no new placements ---
    # Picks up optimizations finished or reset since the previous run and
    # registers newly unique sites. This is what makes a relaunch cheap
    # and what counts crash-recovered clusters without another relaunch.
    harvested = {w: 0 for w in cluster_names}
    if any(len(datasets[w][0].entry_names) for w in cluster_names):
        logger.info(f"{'=' * 80}")
        logger.info("  PHASE 1 — harvesting existing sampling results")
        logger.info(f"{'=' * 80}\n")
        for w in cluster_names:
            ds_smplg, ds_ref = datasets[w]
            if not len(ds_smplg.entry_names):
                continue
            logger.info(f"\n  Cluster {w}:")
            harvested[w] = harvest_cluster(
                client, ds_smplg, ds_ref, opt_lot, args_dict["target_mol"],
                config.rmsd_value, config.rmsd_symmetry, logger,
            )
            refined[w] = len(ds_ref.entry_names)
            logger.info(
                f"  {bcheck} {w}: +{harvested[w]} new unique site(s), "
                f"{refined[w]} refined total"
            )
    # Entries may outlive an interrupted submission or predate the requested
    # refinement LOT. Idempotent submission fills missing records in either case.
    for w in cluster_names:
        if refined[w]:
            _submit_refinement(datasets[w][1])

    count = sum(refined.values())
    logger.info(f"\n  Binding sites after harvest: {count} / target {target}")

    # --- Phase 2: top up with new placements only while below target ---
    if count >= target:
        logger.info(
            f"  Target already met by existing data; no new placements generated.\n"
        )
    else:
        # Never-sampled clusters first: a fresh cluster yields new sites at
        # full rate, a second round on a sampled one mostly rediscovers the
        # minima already filtered out. Sampled clusters are topped up only if
        # the target is still unmet after every cluster has had one round.
        fresh = [w for w in cluster_names if not len(datasets[w][0].entry_names)]
        topup = [w for w in cluster_names if w not in fresh]
        order = [(w, "first round") for w in fresh] + [(w, "top-up") for w in topup]
        for c, (w, pass_kind) in enumerate(order):
            ds_smplg, ds_ref = datasets[w]
            args_dict["cluster"] = qcf.fetch_final_molecule(ds_wc, w, opt_lot)
            args_dict["sampling_opt_dset"] = ds_smplg
            args_dict["refinement_opt_dset"] = ds_ref

            logger.info(f"\n{'=' * 80}")
            logger.info(f"  PHASE 2 — Cluster {c+1}/{len(order)}: {w} ({pass_kind})")
            logger.info(f"  Sampling dataset:    {ds_smplg.name}")
            logger.info(f"  Refinement dataset:  {ds_ref.name}")
            logger.info(f"{'=' * 80}\n")

            debug_path = data_folder / "site_finder" / (str(smol_name) + "_w") / w
            if not debug_path.exists() and config.store_initial_structures:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
            args_dict["debug_path"] = debug_path

            run_sampling(**args_dict)
            _submit_refinement(ds_ref)

            refined[w] = len(ds_ref.entry_names)
            count = sum(refined.values())
            logger.info(
                f"\n  {bcheck} Cluster {w}: {refined[w]} binding sites  |  Running total: {count}"
            )
            if count >= target:
                logger.info(f"\n  Target of {target} binding sites reached. Stopping early.")
                break

    cluster_results = [(w, refined[w]) for w in cluster_names]
    refinement_dsets = [(w, datasets[w][1]) for w in cluster_names if refined[w]]

    # --- Sampling summary (binding sites per cluster, all clusters) ---
    logger.info(f"\n\n{'=' * 80}")
    logger.info(f"  SAMPLING SUMMARY FOR {smol_name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"  {'Cluster':<15} {'Binding sites':>15} {'Harvested':>10}")
    logger.info(f"  {'-'*15} {'-'*15} {'-'*10}")
    for w, n in cluster_results:
        logger.info(f"  {w:<15} {n:>15} {harvested[w]:>10}")
    logger.info(f"  {'-'*15} {'-'*15} {'-'*10}")
    logger.info(f"  {'TOTAL':<15} {count:>15} {sum(harvested.values()):>10}")
    logger.info(f"{'=' * 80}\n")

    # --- Wait for refinement optimizations to finish ---
    REFINEMENT_POLL_FREQUENCY = 120
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  Waiting for refinement optimizations to complete ({ropt_lot})")
    logger.info(f"{'=' * 80}\n")
    refinement_status = []  # [(cluster, n_complete, n_error, n_total)]
    for w, ds_ref in refinement_dsets:
        entries = list(ds_ref.entry_names)
        pids = qcf.get_job_ids(ds_ref, entries, ropt_lot)
        if not pids:
            logger.info(f"  {w}: no refinement records to wait on.")
            refinement_status.append((w, 0, 0, 0))
            continue
        logger.info(f"  {w}: waiting on {len(pids)} refinement opt(s).")
        qcf.wait_for_completion(
            client, pids, REFINEMENT_POLL_FREQUENCY, logger,
        )
        _, counts = qcf.check_for_completion(client, pids)
        n_complete = counts.get("COMPLETE", 0)
        n_error = counts.get("ERROR", 0)
        refinement_status.append((w, n_complete, n_error, len(pids)))
        logger.info(f"  {w}: refinement complete. {bcheck}")

    # --- Refinement summary (terminal status per cluster) ---
    logger.info(f"\n\n{'=' * 80}")
    logger.info(f"  REFINEMENT SUMMARY FOR {smol_name}")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"  {'Cluster':<15} {'Complete':>10} {'Error':>10} {'Total':>10}"
    )
    logger.info(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    tot_c = tot_e = tot_n = 0
    for w, c, e, n in refinement_status:
        logger.info(f"  {w:<15} {c:>10} {e:>10} {n:>10}")
        tot_c += c; tot_e += e; tot_n += n
    logger.info(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    logger.info(
        f"  {'TOTAL':<15} {tot_c:>10} {tot_e:>10} {tot_n:>10}"
    )
    logger.info(f"{'=' * 80}\n")

    logger.removeHandler(file_handler)
    file_handler.close()
