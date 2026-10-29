"""MBE binding-energy assembly workflow (``mbe_extract``).

Ported from beep-mbe's ``assemble_be.run``. Reads a completed ManybodyDataset
and the monomer SinglepointDataset (both read-only) and writes per-site binding
energies plus n-body decomposition tables. Optionally applies a ZPVE correction
borrowed read-only from a prior ``be_hess`` run (see
:class:`beep.models.mbe.MbeZpveConfig`).

No submission or dataset mutation happens here — only ``get_collection`` /
``get_record`` / ``fetch_records`` and, for ZPVE, the read-only helpers in
:mod:`beep.core.mbe_be_tools`.
"""
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import qcelemental

from ..models.mbe import MbeExtractConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import padded_log, beep_banner
from ..core import mbe_be_tools as bt
from ..core.exceptions import MbeExtractError
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient

bcheck = "✔"
gear = "⚙"

welcome_msg = beep_banner(
    "Many-Body Expansion Binding Energy Assembly",
    quote="In nature we never see anything isolated, but everything in connection with something else.",
    quote_author="Johann Wolfgang von Goethe",
    authors="Stefan Vogt-Geisse",
)


def assemble_mbe_be(config: MbeExtractConfig, client: FractalClient, res_folder: Path) -> Path:
    """Assemble BE tables for a completed MBE dataset. Returns the report path."""
    logger = logging.getLogger("beep")

    scheme, pref = bt.resolve_bsse(config.bsse)
    specs = list(config.spec)

    mb_ds_name = bt.manybody_dataset_name(config.dataset, config.opt_level_of_theory)
    sp_ds_name = bt.monomer_dataset_name(config.small_molecule_collection, config.opt_level_of_theory)

    padded_log(logger, "Starting Many-Body Expansion binding energy assembly", padding_char=gear)
    extract_parameters = (
        f"Many-Body Expansion Extraction Parameters:\n"
        f"- Small Molecule: {config.small_molecule}\n"
        f"- Surface Model: {config.surface_model}\n"
        f"- Optimization Level of Theory: {config.opt_level_of_theory}\n"
        f"- ManybodyDataset: {mb_ds_name}\n"
        f"- Monomer SinglepointDataset: {sp_ds_name}\n"
        f"- Specifications: {' '.join(specs)}\n"
        f"- BSSE Scheme: {scheme}\n"
        f"- ZPVE Correction: "
        f"{'enabled (borrowed from be_hess)' if config.zpve and config.zpve.enabled else 'disabled'}\n"
    )
    logger.info(extract_parameters)

    try:
        mb_ds = qcf.get_collection(client, "ManybodyDataset", mb_ds_name)
    except KeyError as exc:
        raise MbeExtractError(f"ManybodyDataset not found: {mb_ds_name}") from exc
    try:
        sp_ds = qcf.get_collection(client, "SinglepointDataset", sp_ds_name)
    except KeyError as exc:
        raise MbeExtractError(f"SinglepointDataset not found: {sp_ds_name}") from exc
    logger.info(f"Using ManybodyDataset: {mb_ds_name}")
    logger.info(f"Using SinglepointDataset: {sp_ds_name}")

    all_entry_names = bt.dataset_entry_names(mb_ds)
    if config.surface_model not in all_entry_names:
        raise MbeExtractError(
            f"Surface entry '{config.surface_model}' not found in ManybodyDataset "
            f"'{mb_ds_name}'. BE decomposition requires surface MBE components."
        )

    entry_names = bt.resolve_entries(mb_ds, config.entries, config.surface_model)
    if not entry_names:
        raise MbeExtractError("No supermolecule entries available for BE assembly.")

    fetch_entries = list(dict.fromkeys([config.surface_model, *entry_names]))
    try:
        mb_ds.fetch_records(entry_names=fetch_entries, specification_names=specs, include=["properties"])
    except Exception as exc:
        logger.debug(f"Manybody fetch_records failed: {exc}")
    sp_specs = [f"monomer_{spec}" for spec in specs]
    try:
        sp_ds.fetch_records(entry_names=[config.small_molecule], specification_names=sp_specs, include=["properties"])
    except Exception as exc:
        logger.debug(f"Singlepoint fetch_records failed: {exc}")

    padded_log(logger, "Computing binding energies from MBE components", padding_char=gear)
    hartree_to_kcal = qcelemental.constants.hartree2kcalmol

    total_data: Dict[str, Dict[str, float]] = {entry: {} for entry in entry_names}
    decomp_by_spec: Dict[str, Dict[str, Dict[str, float]]] = {}
    contrib_by_spec: Dict[str, Dict[str, Dict[str, float]]] = {}
    conv_by_spec: Dict[str, Dict[str, Dict]] = {}

    for spec in specs:
        surface_rec = mb_ds.get_record(entry_name=config.surface_model, specification_name=spec)
        if surface_rec is None:
            raise MbeExtractError(
                f"Surface entry '{config.surface_model}' has no record under spec '{spec}'."
            )
        surface_components = bt.extract_mbe_components(surface_rec, pref)

        sp_spec_name = f"monomer_{spec}"
        monomer_rec = sp_ds.get_record(entry_name=config.small_molecule, specification_name=sp_spec_name)
        if monomer_rec is None:
            raise MbeExtractError(
                f"Monomer record not found in '{sp_ds_name}' for '{config.small_molecule}' / '{sp_spec_name}'."
            )
        monomer_energy = bt.extract_monomer_energy(monomer_rec)

        decomp_by_spec[spec] = {}
        contrib_by_spec[spec] = {}
        conv_by_spec[spec] = {}

        for entry in entry_names:
            rec = mb_ds.get_record(entry_name=entry, specification_name=spec)
            super_components = (
                bt.extract_mbe_components(rec, pref)
                if rec else {"e1": None, "e2": None, "e3": None, "etot": None}
            )
            be_values = bt.compute_be_values(super_components, surface_components, monomer_energy)
            be_total = be_values["be_total"]
            total_data[entry][spec] = float("nan") if be_total is None else float(be_total * hartree_to_kcal)

            be_values_kcal = {
                key: (None if value is None else value * hartree_to_kcal)
                for key, value in be_values.items()
            }
            decomp_by_spec[spec][entry] = bt.build_cumulative(be_values_kcal)
            contrib_by_spec[spec][entry] = bt.build_contributions(be_values_kcal)
            conv_by_spec[spec][entry] = bt.compute_convergence(
                be_values_kcal, config.convergence_tol
            )

    df_total_be = pd.DataFrame.from_dict(total_data, orient="index", columns=specs).reindex(entry_names)
    # Union of per-site orders, sorted numerically (including orders >= 10).
    # Lower-order sites retain NaN in columns reported only by other sites.
    df_decomp, df_contrib = {}, {}
    for spec in specs:
        columns = sorted({col for row in decomp_by_spec[spec].values() for col in row},
                         key=lambda col: int(col[3:-1]))
        df_decomp[spec] = pd.DataFrame.from_dict(decomp_by_spec[spec], orient="index").reindex(
            index=entry_names, columns=columns)
        df_contrib[spec] = pd.DataFrame.from_dict(contrib_by_spec[spec], orient="index").reindex(
            index=entry_names, columns=[f"{col}_contrib" for col in columns])

    # --- MBE truncation-error / convergence estimate (symmetric geometric bar) ---
    conv_columns = ["BE_total", "n_body_max", "delta_last", "ratio_r",
                    "error_bar", "rel_error", "converged"]
    df_conv = {}
    for spec in specs:
        rows = {}
        for entry in entry_names:
            c = conv_by_spec[spec][entry]
            rows[entry] = {
                "BE_total": total_data[entry].get(spec, float("nan")),
                "n_body_max": c["n_body_max"],
                "delta_last": c["delta_last"],
                "ratio_r": c["ratio_r"],
                "error_bar": c["error_bar"],
                "rel_error": c["rel_error"],
                "converged": c["converged"],
            }
        df_conv[spec] = pd.DataFrame.from_dict(rows, orient="index", columns=conv_columns).reindex(entry_names)

    padded_log(logger, "Saving all dataframes to CSV", padding_char=gear)
    data_dir = res_folder / "be_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df_total_be.to_csv(data_dir / "total_be.csv", index=True, float_format="%.8f")
    for spec, df in df_decomp.items():
        safe_spec = bt.safe_filename(spec)
        df.to_csv(data_dir / f"decomp__{safe_spec}.csv", index=True, float_format="%.8f")
        df_contrib[spec].to_csv(data_dir / f"contrib__{safe_spec}.csv", index=True, float_format="%.8f")
        df_conv[spec].to_csv(data_dir / f"convergence__{safe_spec}.csv", index=True, float_format="%.8f")

    # --- optional ZPVE correction (read-only borrow from be_hess) ---
    df_total_be_zpve = None
    if config.zpve is not None and config.zpve.enabled:
        zpve_molecule = config.zpve.molecule or config.small_molecule
        padded_log(logger, "Applying ZPVE correction borrowed from be_hess", padding_char=gear)
        logger.info(f"Borrowing ZPVE corrections from be_hess for {zpve_molecule}.")
        delta_zpve = bt.borrow_zpve_corrections(
            client,
            molecule=zpve_molecule,
            hessian_clusters=config.zpve.hessian_clusters,
            opt_lot=config.opt_level_of_theory,
            entry_names=entry_names,
            scale_factor=config.zpve.scale_factor,
            imag_threshold=config.zpve.imag_threshold,
            logger=logger,
        )
        # Sign convention: MBE BEs are POSITIVE (fragments minus complex) and
        # Delta_ZPVE = ZPVE(complex) - ZPVE(parts) is also positive, so the
        # ZPVE-corrected BE is BE - Delta_ZPVE. This differs from
        # workflows/extract.py, where BEs are NEGATIVE-convention and the
        # correction is added (BE + Delta_ZPVE).
        df_total_be_zpve = pd.DataFrame(index=entry_names)
        for spec in specs:
            df_total_be_zpve[f"{spec}+ZPVE"] = df_total_be[spec] - delta_zpve
        df_total_be_zpve["Delta_ZPVE"] = delta_zpve
        df_total_be_zpve.to_csv(data_dir / "total_be_zpve.csv", index=True, float_format="%.8f")

    # --- text report ---
    lines: List[str] = []
    lines.append("BEEP many-body expansion binding energy report")
    lines.append(f"Molecule: {config.small_molecule}")
    lines.append("Units: kcal/mol")
    lines.append(f"BSSE scheme: {scheme}")
    lines.append(f"Specifications: {', '.join(specs)}")
    lines.append(f"CSV output directory: {data_dir}")
    lines.append("")
    lines.append("Total binding energies (kcal/mol)")
    lines.append(bt.render_table(df_total_be))
    if df_total_be_zpve is not None:
        lines.append("")
        lines.append("Total binding energies with ZPVE correction (kcal/mol)")
        lines.append(bt.render_table(df_total_be_zpve))
    for spec in specs:
        lines.append("")
        lines.append(f"Decomposition (cumulative) - spec={spec}")
        lines.append(bt.render_table(df_decomp[spec]))
        lines.append("")
        lines.append(f"Per-body contributions - spec={spec}")
        lines.append(bt.render_table(df_contrib[spec]))
        lines.append("")
        lines.append(f"MBE truncation error (geometric tail, +/- kcal/mol) - spec={spec}")
        conv_rows = {
            e: {**conv_by_spec[spec][e], "BE_total": total_data[e].get(spec)}
            for e in entry_names
        }
        lines.append(bt.format_convergence_table(conv_rows))

    padded_log(logger, "Summary of MBE binding energy results", padding_char=gear)
    logger.info("\nTotal binding energies (kcal/mol)\n" + bt.render_table(df_total_be))
    if df_total_be_zpve is not None:
        logger.info(
            "\nTotal binding energies with ZPVE correction (kcal/mol)\n"
            + bt.render_table(df_total_be_zpve)
        )
    for spec in specs:
        conv_rows = {
            e: {**conv_by_spec[spec][e], "BE_total": total_data[e].get(spec)}
            for e in entry_names
        }
        logger.info(
            f"\nMBE truncation error (+/- kcal/mol) - spec={spec}\n"
            + bt.format_convergence_table(conv_rows)
        )

    report_path = res_folder / f"mbe_extract_{config.small_molecule}.out"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"\nWrote BE report to {report_path} {bcheck}")
    return report_path


def run(config: MbeExtractConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    res_folder = Path.cwd() / config.small_molecule
    res_folder.mkdir(parents=True, exist_ok=True)

    log_file = res_folder / f"mbe_extract_{config.small_molecule}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    config_path = res_folder / f"mbe_extract_{config.small_molecule}.json"
    config_path.write_text(safe_config_dump(config))

    logger.info(welcome_msg)
    try:
        assemble_mbe_be(config, client, res_folder)
        logger.info(
            "\nThank you for using the many-body expansion binding energy assembly suite!"
        )
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()
