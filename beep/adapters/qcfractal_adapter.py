"""
QCFractal Adapter — single module encapsulating ALL QCFractal server I/O.

Updated for QCPortal v0.63+ API.

Every function that touches the QCFractal server lives here. The rest of the
BEEP codebase (core/, workflows/) calls these functions instead of importing
qcportal directly.

Functions are organized by category:
  - Connection
  - Collection management
  - Molecule queries
  - Job submission
  - Job monitoring
  - Data retrieval
  - RMSD filter (server-coupled version)
  - ZPVE (server-coupled version)
"""
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from typing import Callable, Sequence  # MBE monitoring helpers (additive)
from collections import Counter
from dataclasses import dataclass

from qcportal import PortalClient, PortalRequestError
from qcportal.record_models import RecordStatusEnum, PriorityEnum
from qcportal.singlepoint.record_models import QCSpecification, SinglepointDriver
from qcportal.optimization.record_models import OptimizationSpecification
from qcportal.optimization.dataset_models import OptimizationDataset
from qcportal.singlepoint.dataset_models import (
    SinglepointDataset, SinglepointDatasetNewEntry,
)
from qcportal.reaction.dataset_models import ReactionDataset
from qcportal.manybody import (
    ManybodyDataset, ManybodyKeywords, ManybodySpecification,
)
import numpy as np
from qcportal.reaction.record_models import ReactionSpecification, ReactionKeywords
from qcelemental.models.molecule import Molecule
from pydantic import ValidationError

from ..core.logging_utils import log_formatted_list, padded_log
from ..core.stoichiometry import be_stoichiometry
from ..core.errors import DatasetNotFound, LevelOfTheoryNotFound
from ..models.base import split_lot_string

# Backward-compatible aliases
FractalClient = PortalClient
Dataset = SinglepointDataset

# Re-export types so that workflows can import them from the adapter
# instead of reaching into qcportal internals directly.
__all__ = [
    "FractalClient", "PortalClient", "Dataset", "SinglepointDataset",
    "OptimizationDataset", "ReactionDataset", "Molecule",
    "RecordStatusEnum",
    "is_complete", "is_incomplete", "is_error", "status_label",
]
# MBE / manybody exports (additive; see the MBE helper section below).
__all__ += ["ManybodyDataset", "get_or_create_manybody_dataset",
            "wait_for_manybody_completion", "wait_for_dataset_records",
            "ManybodyMonitorResult"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Default upper bound for the job-wait helpers below. Batches can sit queued
# behind other work on a shared manager for days, so keep this generous.
DEFAULT_MAX_WAIT = 14 * 24 * 3600  # two weeks

STOICH_TYPES = ("bsse", "be_nocp", "ie", "ie_nocp", "de")
# Stoichiometries saved for MACE (MLP) runs: only ghost-free ones. The
# counterpoise stoichiometries (``bsse``, ``ie``) use ghost atoms, which a
# basis-set-free potential cannot evaluate (the harness treats ghosts as real
# atoms, yielding garbage). ``ie_nocp`` = ``be_nocp - de`` replaces ``ie``.
MACE_STOICH_TYPES = ("be_nocp", "ie_nocp", "de")


_COLLECTION_TYPE_MAP = {
    "OptimizationDataset": "optimization",
    "ReactionDataset": "reaction",
    "Dataset": "singlepoint",
    "SinglepointDataset": "singlepoint",
    "ManybodyDataset": "manybody",
}


def _stoich_dataset_name(base_name: str, stoich_type: str) -> str:
    """Build the dataset name for a given stoichiometry type."""
    return f"{base_name}_{stoich_type}"


def _resolve_dataset_type(collection_type):
    """Convert old collection_type string or class to v0.63 dataset_type string."""
    if isinstance(collection_type, str):
        return _COLLECTION_TYPE_MAP.get(collection_type, collection_type.lower())
    name = getattr(collection_type, "__name__", str(collection_type))
    return _COLLECTION_TYPE_MAP.get(name, name.lower())


def _check_insert_meta(meta, what: str, logger: Optional[logging.Logger] = None) -> bool:
    """Inspect the ``InsertMetadata`` returned by ``add_specification`` /
    ``add_entry`` / ``add_entries`` and warn when the server rejected (part
    of) the insertion.

    qcportal's dataset-level add calls are idempotent for identical
    content, but a same-name specification or entry whose *content* differs
    from what the server already holds is reported through
    ``meta.errors`` (``meta.success`` is False) rather than raised. Silently
    discarding that metadata means the old spec keeps running under the new
    name. Returns True when the insertion was fully accepted.
    """
    if meta is None:
        return True
    success = getattr(meta, "success", None)
    if success is not False:
        return True
    log = logger or logging.getLogger("beep")
    n_existing = getattr(meta, "n_existing", 0)
    n_inserted = getattr(meta, "n_inserted", 0)
    errors = getattr(meta, "error_string", None) or str(getattr(meta, "errors", ""))
    log.warning(
        f"Server did not fully accept {what}: inserted={n_inserted}, "
        f"existing={n_existing}. This usually means an item with the same "
        f"name already exists with different content and was NOT updated. "
        f"Errors: {errors.strip()}"
    )
    return False


def is_complete(status):
    """Check if a record status represents completion."""
    return status == RecordStatusEnum.complete


def is_incomplete(status):
    """Check if a record status represents a running/waiting state."""
    return status in (RecordStatusEnum.running, RecordStatusEnum.waiting)


def is_error(status):
    """Check if a record status represents an error."""
    return status == RecordStatusEnum.error


def status_label(status):
    """Map v0.63 RecordStatusEnum to old-style uppercase string."""
    if is_complete(status):
        return "COMPLETE"
    if is_incomplete(status):
        return "INCOMPLETE"
    if is_error(status):
        return "ERROR"
    return str(status.value).upper()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(address: str = "localhost:7777", username: str = None,
            password: str = None, verify: bool = False) -> PortalClient:
    """Create and return a PortalClient connection.

    ``verify`` defaults to False because BEEP deployments typically target
    QCFractal servers on internal HTTP-only addresses or reverse-proxied
    with TLS terminated upstream. Pass ``verify=True`` for HTTPS servers
    with a publicly-trusted certificate. See ``models.base.ServerConfig``.
    """
    return PortalClient(
        address=address, verify=verify,
        username=username, password=password,
        show_motd=False,
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_collection(client: PortalClient, collection_type: str, name: str):
    """Retrieve a dataset by type and name.

    Raises KeyError if the dataset does not exist (preserves v0.15 behavior).
    """
    ds_type = _resolve_dataset_type(collection_type)
    try:
        return client.get_dataset(ds_type, name)
    except (KeyError, PortalRequestError) as e:
        raise KeyError(
            f"Dataset '{name}' of type '{ds_type}' not found"
        ) from e


def get_or_create_opt_dataset(client: PortalClient, name: str):
    """Get an existing OptimizationDataset or create a new one."""
    try:
        return client.get_dataset("optimization", name)
    except (KeyError, PortalRequestError):
        return client.add_dataset("optimization", name)


def create_reaction_dataset(client: PortalClient, name: str,
                            program: str = "psi4"):
    """Return an existing ReactionDataset by ``name`` or create a fresh one.

    Idempotent: when the dataset already exists on the server it is returned
    unchanged. Previously this helper deleted and recreated the dataset on
    every call, which wiped any specs and entries registered by earlier
    runs. qcportal 0.63+ makes spec / entry registration idempotent at the
    dataset level, so wiping is no longer needed.
    """
    try:
        return client.get_dataset("reaction", name)
    except (KeyError, PortalRequestError):
        return client.add_dataset("reaction", name)


def delete_collection(client: PortalClient, collection_type: str,
                      name: str) -> None:
    """Delete a dataset, ignoring errors if it doesn't exist."""
    try:
        ds_type = _resolve_dataset_type(collection_type)
        ds = client.get_dataset(ds_type, name)
        client.delete_dataset(ds.id, delete_records=False)
    except (KeyError, PortalRequestError):
        pass


def check_collection_exists(client: PortalClient, collection_type: str,
                            name: str) -> bool:
    """Check whether a dataset exists on the server."""
    ds_type = _resolve_dataset_type(collection_type)
    try:
        client.get_dataset(ds_type, name)
        return True
    except (KeyError, PortalRequestError):
        return False


def check_collection_existence(client: PortalClient, *collections,
                                collection_type: str = "OptimizationDataset") -> None:
    """Validate that all named datasets exist. Raises DatasetNotFound if any is missing."""
    logger = logging.getLogger("beep")
    for collection in collections:
        if not check_collection_exists(client, collection_type, collection):
            raise DatasetNotFound(
                f"Collection {collection} does not exist. "
                "Please create it first. Exiting..."
            )
        ds_type = _resolve_dataset_type(collection_type)
        logger.info(f"The {ds_type} dataset named {collection} exists \u2714")


def get_or_create_collection(client: PortalClient, name: str,
                             collection_type=OptimizationDataset):
    """Get or create a dataset of any type."""
    logger = logging.getLogger("beep")
    ds_type = _resolve_dataset_type(collection_type)
    type_name = (collection_type.__name__
                 if hasattr(collection_type, "__name__")
                 else str(collection_type))
    try:
        ds = client.get_dataset(ds_type, name)
        logger.info(
            f"Collection of type {type_name} with name {name} "
            "already exists. \u2714\n"
        )
    except (KeyError, PortalRequestError):
        ds = client.add_dataset(ds_type, name)
        logger.info(f"Creating new {type_name}: {name}.\n")
    return ds


def check_optimized_molecule(ds, opt_lot: str, mol_names) -> None:
    """Check that all named molecules have COMPLETE optimization records."""
    if isinstance(mol_names, str):
        mol_names = [mol_names]
    for mol in list(mol_names):
        # fetch_opt_record translates both "entry not in dataset" (HTTP 400
        # from the server) and "no record at this spec" into KeyError, so
        # callers get LevelOfTheoryNotFound instead of a raw
        # PortalRequestError for a missing entry.
        try:
            record = fetch_opt_record(ds, mol, opt_lot)
        except KeyError as e:
            raise LevelOfTheoryNotFound(
                f"{opt_lot} level of theory for {mol} or the entry itself "
                f"does not exist in {ds.name} dataset. "
                "Add the molecule and optimize it first\n"
            ) from e
        if is_incomplete(record.status):
            raise ValueError(
                f" Optimization has status {record.status} restart it or wait"
            )
        elif is_error(record.status):
            raise ValueError(
                f" Optimization has status {record.status} restart it or wait"
            )


def get_molecular_multiplicity(client: PortalClient, dataset,
                               molecule_name: str) -> int:
    """Get spin multiplicity from the initial molecule of a dataset entry."""
    entry = None
    for name_variant in [molecule_name, molecule_name.upper(), molecule_name.lower()]:
        try:
            entry = dataset.get_entry(name_variant)
            if entry is not None:
                break
        except Exception:
            continue
    if entry is None:
        raise KeyError(f"Entry '{molecule_name}' not found in dataset")
    return entry.initial_molecule.molecular_multiplicity


def get_xyz(client: PortalClient, dataset: str, mol_name: str,
            level_theory: str,
            collection_type: str = "OptimizationDataset") -> str:
    """Fetch optimized geometry as XYZ string."""
    ds_opt = get_collection(client, collection_type, dataset)
    record = fetch_opt_record(ds_opt, mol_name, level_theory)
    mol = record.final_molecule
    geom = mol.to_string(dtype="xyz")
    xyz_list = geom.splitlines()[2:]
    return "\n".join(xyz_list)


def wait_for_dataset_completion(client: PortalClient, ds_opt, opt_lot: str,
                                 logger: logging.Logger,
                                 wait_interval: int = 600) -> None:
    """Poll a dataset until all records for a given LOT are COMPLETE."""
    while True:
        incomplete = []
        n_complete = 0
        n_error = 0
        for entry_name in ds_opt.entry_names:
            try:
                record = ds_opt.get_record(entry_name, opt_lot)
                if record is not None:
                    if is_incomplete(record.status):
                        incomplete.append(entry_name)
                    elif is_error(record.status):
                        n_error += 1
                        logger.info(
                            f"Warning: Entry '{entry_name}' finished with ERROR"
                        )
                    elif is_complete(record.status):
                        n_complete += 1
            except (KeyError, TypeError):
                continue

        n_total = n_complete + len(incomplete) + n_error
        if not incomplete:
            logger.info(
                f"Dataset '{ds_opt.name}': {n_complete}/{n_total} COMPLETE"
                + (f", {n_error} ERROR" if n_error else "")
                + ". All done."
            )
            return

        logger.info(
            f"Dataset '{ds_opt.name}': {n_complete}/{n_total} COMPLETE, "
            f"{len(incomplete)} incomplete"
            + (f", {n_error} ERROR" if n_error else "")
            + f". Waiting {wait_interval}s..."
        )
        time.sleep(wait_interval)
        # Re-fetch to get updated statuses
        ds_opt = get_collection(client, "OptimizationDataset", ds_opt.name)


# ---------------------------------------------------------------------------
# Molecule queries
# ---------------------------------------------------------------------------

def fetch_atom_molecule(client: PortalClient, atoms_collection: str,
                        atom_name: str) -> Molecule:
    """Fetch an atom's molecule from a SinglepointDataset.

    Atoms (single-atom species) are stored in a dedicated SinglepointDataset
    rather than an OptimizationDataset since they cannot be optimized.

    Raises KeyError when ``atom_name`` is not in ``atoms_collection``. Other
    PortalRequestErrors (server / network / auth failures) propagate
    unchanged — see ``fetch_opt_record`` for the rationale.
    """
    ds = client.get_dataset("singlepoint", atoms_collection)
    try:
        entry = ds.get_entry(atom_name)
    except PortalRequestError as e:
        msg = str(e)
        if "Missing" in msg and "entries" in msg:
            raise KeyError(
                f"Atom '{atom_name}' not found in singlepoint dataset "
                f"'{atoms_collection}'"
            ) from e
        raise
    if entry is None:
        raise KeyError(
            f"Atom '{atom_name}' not found in singlepoint dataset "
            f"'{atoms_collection}'"
        )
    return entry.molecule


def fetch_molecules(client: PortalClient, mol_ids) -> List[Molecule]:
    """Query molecules by ID(s). Always returns a list."""
    if not isinstance(mol_ids, (list, tuple)):
        mol_ids = [mol_ids]
    result = client.get_molecules(mol_ids)
    if not isinstance(result, list):
        return [result]
    return result


def fetch_opt_record(ds_opt, entry_name: str, opt_lot: str):
    """Get the optimization record for an entry.

    Raises KeyError when the entry doesn't exist in the dataset or when
    the record for the requested specification is missing. Other
    PortalRequestErrors (server/network/auth failures) propagate
    unchanged — callers should treat them as transient errors, not as
    "entry not found".
    """
    try:
        record = ds_opt.get_record(entry_name, opt_lot)
    except PortalRequestError as e:
        # qcportal returns HTTP 400 "Missing N entries: ..." when the
        # entry name is not in the dataset. Translate that one specific
        # case to KeyError so callers can distinguish "entry not in
        # dataset" from "server / network failure". The string match
        # is fragile to qcportal message changes; revisit if the wording
        # shifts upstream.
        msg = str(e)
        if "Missing" in msg and "entries" in msg:
            raise KeyError(
                f"Entry '{entry_name}' not found in dataset"
            ) from e
        raise
    if record is None:
        raise KeyError(
            f"No record for entry '{entry_name}' with specification '{opt_lot}'"
        )
    return record


def fetch_final_molecule(ds_opt, entry_name: str, opt_lot: str) -> Molecule:
    """Get the final optimized molecule from an optimization record."""
    return fetch_opt_record(ds_opt, entry_name, opt_lot).final_molecule


def fetch_initial_molecule(ds_opt, entry_name: str, opt_lot: str) -> Molecule:
    """Get the initial molecule from an optimization record."""
    return fetch_opt_record(ds_opt, entry_name, opt_lot).initial_molecule


def fetch_opt_cell(ds_opt, entry_name: str, opt_lot: str):
    """The periodic cell an optimization was run under, from its QC spec keywords.

    sampling_periodic passes ``cell``/``pbc`` through the singlepoint keywords of
    the optimization specification, and QCFractal does not propagate molecule
    extras into the optimized output geometry. Reading the cell back from the
    record is therefore the only source that is guaranteed to match the geometry
    being evaluated. Returns ``(cell, pbc)``, either of which may be None.
    """
    record = fetch_opt_record(ds_opt, entry_name, opt_lot)
    keywords = dict(record.specification.qc_specification.keywords or {})
    return keywords.get("cell"), keywords.get("pbc")


def fetch_entry_initial_molecule(ds_opt, entry_name: str) -> Molecule:
    """Initial molecule of an OptimizationDataset *entry*, independent of any spec.

    The input geometry of an entry is a property of the entry, not of a
    calculation run on it, so retrieving it must not require a matching
    optimization specification to exist. Used to read bare slabs out of a
    surface collection whose registered specs belong to a different adsorbate.
    """
    return ds_opt.get_entry(entry_name).initial_molecule


def fetch_opt_molecules(ds_opt, entry_list: List[str], opt_lot: str,
                        status: str = "COMPLETE") -> List[Tuple[str, Molecule]]:
    """
    Retrieve final molecules from an optimization dataset for entries with
    the specified status.
    """
    target_status = RecordStatusEnum(status.lower())
    mol_list = []
    for n in entry_list:
        record = ds_opt.get_record(n, opt_lot, force_refetch=True)
        if record is not None and record.status == target_status:
            mol_list.append((n, record.final_molecule))
    return mol_list


def fetch_opt_energies(ds_opt, entry_list: List[str], opt_lot: str,
                       status: str = "COMPLETE") -> Dict[str, float]:
    """Final energies from an optimization dataset, keyed by entry name.

    Used by the periodic duplicate filter to keep the lowest-energy member of each
    group of equivalent sites rather than whichever was encountered first.
    """
    target_status = RecordStatusEnum(status.lower())
    energies: Dict[str, float] = {}
    for n in entry_list:
        record = ds_opt.get_record(n, opt_lot, force_refetch=True)
        if record is not None and record.status == target_status and record.energies:
            energies[n] = float(record.energies[-1])
    return energies


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def _require_keywords_dict(keywords, what: str) -> dict:
    """Return ``keywords`` as a dict, or raise TypeError for anything else.

    QCFractal 0.15 stored keywords server-side under integer IDs; qcportal
    0.63+ takes inline dicts. Previously a non-dict value was silently
    replaced by ``{}``, so user options vanished without a trace.
    """
    if keywords is None:
        return {}
    if isinstance(keywords, dict):
        return keywords
    raise TypeError(
        f"{what} must be a dict of option name -> value (or null), got "
        f"{type(keywords).__name__} {keywords!r}. QCFractal keyword IDs are "
        "no longer supported; write the keywords inline, e.g. "
        '{"scf_type": "df"}.'
    )


def add_opt_specification(ds_opt, spec_dict: dict,
                          overwrite: bool = True) -> None:
    """Add an optimization specification to a dataset.

    Accepts old-style spec dicts with 'name', 'description',
    'optimization_spec', and 'qc_spec' keys and translates to v0.63
    OptimizationSpecification objects.

    The ``overwrite`` parameter is accepted for backward compatibility but
    is ignored — v0.63 ``add_specification`` is idempotent (silently
    reports existing specs without error).
    """
    name = spec_dict.get("name", "default").lower()
    description = spec_dict.get("description", "")
    qc_spec_dict = spec_dict.get("qc_spec", {})
    opt_spec_dict = spec_dict.get("optimization_spec", {})

    # In v0.63, keywords are dicts, not server-stored IDs. A non-dict
    # (e.g. a legacy integer keyword ID) used to be silently replaced by {}.
    qc_keywords = _require_keywords_dict(
        qc_spec_dict.get("keywords"), f"qc_spec keywords of spec '{name}'",
    )

    qc_spec = QCSpecification(
        program=qc_spec_dict.get("program", "psi4"),
        driver=SinglepointDriver.deferred,
        method=qc_spec_dict.get("method", ""),
        basis=qc_spec_dict.get("basis"),
        keywords=qc_keywords,
    )

    opt_keywords = _require_keywords_dict(
        opt_spec_dict.get("keywords"), f"optimization keywords of spec '{name}'",
    )

    opt_spec = OptimizationSpecification(
        program=opt_spec_dict.get("program", "geometric"),
        qc_specification=qc_spec,
        keywords=opt_keywords,
    )

    meta = ds_opt.add_specification(name, opt_spec, description=description)
    _check_insert_meta(meta, f"optimization specification '{name}' in {ds_opt.name}")


def add_opt_entry(ds_opt, name: str, molecule: Molecule,
                  save: bool = True) -> None:
    """Add an entry to an optimization dataset.

    The ``save`` parameter is accepted for backward compatibility but
    is ignored — v0.63 entries are saved immediately via the API.
    """
    meta = ds_opt.add_entry(name=name, initial_molecule=molecule)
    _check_insert_meta(meta, f"entry '{name}' in {ds_opt.name}")


def submit_optimizations(ds_opt, opt_lot: str, tag: str, subset=None):
    """Submit optimization computations."""
    entry_names = list(subset) if subset else None
    return ds_opt.submit(
        entry_names=entry_names,
        specification_names=[opt_lot],
        compute_tag=tag,
    )


def _energy_spec_name(method: str, basis: Optional[str],
                      spec_name: Optional[str] = None) -> str:
    """Spec name used by :func:`submit_energies` for an energy computation.

    Single source of truth so that the monitoring side
    (:func:`_collect_reaction_record_ids`) can restrict itself to exactly the
    specs submitted in the same call.
    """
    if spec_name is None:
        spec_name = f"{method}_{basis}" if basis else method
    return spec_name.lower()


def submit_energies(client: PortalClient, rdset_base_name: str,
                    method: str, basis: Optional[str], program: str,
                    stoich: str, tag: str, keywords=None,
                    spec_name: Optional[str] = None):
    """Submit energy computations to a stoichiometry-specific ReactionDataset.

    If ``basis`` is ``None`` the spec is named after the method alone (used for
    bare dispersion specs like ``pbe-d3bj`` that have no basis set).
    ``spec_name`` overrides the derived name — used when ``method`` is not a
    presentable label (e.g. a MACE model file path, whose spec is named by
    the model alias instead).
    """
    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)
    spec_name = _energy_spec_name(method, basis, spec_name)

    kw_dict = _require_keywords_dict(keywords, f"keywords of energy spec '{spec_name}'")
    qc_spec = QCSpecification(
        program=program,
        driver=SinglepointDriver.energy,
        method=method,
        basis=basis,
        keywords=kw_dict,
    )
    rxn_spec = ReactionSpecification(
        program="reaction",
        singlepoint_specification=qc_spec,
        keywords=ReactionKeywords(),
    )
    meta = ds.add_specification(spec_name, rxn_spec)
    _check_insert_meta(meta, f"reaction specification '{spec_name}' in {ds_name}")

    return ds.submit(
        specification_names=[spec_name],
        compute_tag=tag,
    )


def submit_hessians(client: PortalClient, program: str, method: str,
                    basis: str, mol_ids: list, keywords, tag: str):
    """Submit hessian computations via client.add_singlepoints."""
    kw_dict = _require_keywords_dict(keywords, "Hessian keywords")
    return client.add_singlepoints(
        molecules=mol_ids,
        program=program,
        driver="hessian",
        method=method,
        basis=basis,
        keywords=kw_dict,
        compute_tag=tag,
    )


def submit_gcp_corrections(client: PortalClient, mol_ids: list, tag: str):
    """Submit gCP corrections at dft/def2-tzvp via the mctc-gcp program.

    The gCP correction depends on the basis set and SCF type only, not on
    the specific DFT functional, so one record per (molecule, basis) is
    enough — the result is reused across every gcp-compatible functional
    at extract time. We currently only target ``dft/def2-tzvp``; expand
    here if BEEP adopts additional basis sets in the gCP-parametrised
    range.
    """
    return client.add_singlepoints(
        molecules=mol_ids,
        program="mctc-gcp",
        driver="energy",
        method="dft/def2-tzvp",
        basis=None,
        keywords={},
        compute_tag=tag,
    )


def fetch_gcp_corrections(
    client: PortalClient, mol_ids: List[int],
) -> Dict[int, float]:
    """Fetch gCP correction energies (hartree) for the given molecule IDs.

    Pairs with ``submit_gcp_corrections``. Returns ``{mol_id → gcp_energy}``
    for records that are complete; molecules with missing/incomplete
    records are silently omitted (caller should treat them as missing).
    """
    results = client.query_singlepoints(
        program="mctc-gcp",
        method="dft/def2-tzvp",
        molecule_id=mol_ids,
    )
    out: Dict[int, float] = {}
    for r in results:
        if not is_complete(r.status):
            continue
        # mctc-gcp publishes the correction as ``return_result`` (plus a few
        # human-readable aliases like ``"gcp correction energy"``); the
        # ``return_energy`` key Psi4/QC backends use does not exist here.
        e = r.return_result
        if e is not None:
            out[r.molecule_id] = float(e)
    return out


def add_reaction(client: PortalClient, rdset_base_name: str,
                 name: str, stoichiometry: dict) -> None:
    """Add reaction entries across stoichiometry-specific datasets.

    ``stoichiometry`` is a dict mapping stoich type names to lists of
    ``(Molecule, coefficient)`` tuples (as returned by ``be_stoichiometry``).
    Each stoich type is stored in its own ReactionDataset named
    ``{rdset_base_name}_{stoich_type}``.
    """
    for stoich_type, mol_coeff_list in stoichiometry.items():
        ds_name = _stoich_dataset_name(rdset_base_name, stoich_type)
        ds = client.get_dataset("reaction", ds_name)
        # be_stoichiometry returns (Molecule, coeff); v0.63 wants (coeff, Molecule)
        stoichiometries = [(coeff, mol) for mol, coeff in mol_coeff_list]
        meta = ds.add_entry(name=name, stoichiometries=stoichiometries)
        _check_insert_meta(meta, f"reaction entry '{name}' in {ds_name}")


def add_keywords(client: PortalClient, keyword_set) -> Any:
    """In v0.63 keywords are inline in specifications — returns the dict."""
    if isinstance(keyword_set, dict):
        return keyword_set
    return getattr(keyword_set, "values", keyword_set)


def create_keyword_set(values: dict) -> dict:
    """In v0.63, keywords are plain dicts. Returns values directly."""
    return values


def query_keywords(client: PortalClient):
    """Not available in v0.63 — keywords are inline in specifications."""
    raise NotImplementedError(
        "query_keywords is not available in QCPortal v0.63+. "
        "Keywords are now inline in specifications."
    )


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def get_job_ids(ds_opt, entry_list: List[str], opt_lot: str) -> List[int]:
    """Get record IDs for optimization entries.

    Names not present in the dataset are skipped silently; the server
    raises HTTP 400 on missing entries and we don't want callers to
    crash when they ask for names that were never inserted (e.g. when
    pose generation returns fewer structures than requested).
    """
    known = set(ds_opt.entry_names)
    pid_list = []
    for n in entry_list:
        if n not in known:
            continue
        record = ds_opt.get_record(n, opt_lot)
        if record is not None:
            pid_list.append(record.id)
    return pid_list


def check_for_completion(client: PortalClient, pid: List[int],
                         frequency: int = 60) -> Tuple[bool, Counter]:
    """
    Check job completion status. Returns (all_complete, status_counts).

    Status counts use old-style uppercase strings for backward compatibility.
    """
    status_labels = []
    if pid:
        records = client.get_records(pid, missing_ok=True)
        if not isinstance(records, list):
            records = [records]
        for r in records:
            if r is not None:
                status_labels.append(status_label(r.status))
    status_counts = Counter(status_labels)
    all_jobs_complete = "INCOMPLETE" not in status_labels
    return all_jobs_complete, status_counts


def wait_for_completion(client: PortalClient, pid_list: List[int],
                        frequency: int, logger: logging.Logger,
                        max_wait: int = DEFAULT_MAX_WAIT) -> None:
    """Poll until all jobs reach a terminal state (complete or error).

    Parameters
    ----------
    max_wait : int
        Maximum total wait time in seconds (default two weeks).
        Raises ``TimeoutError`` if exceeded.
    """
    if not pid_list:
        return

    logger.info("Checking for job completion....")
    elapsed = 0
    while True:
        jobs_complete, counts = check_for_completion(client, pid_list, frequency)
        status_str = " ".join(
            f"{s}: {count}" for s, count in counts.items()
        )
        logger.info(f"The status of the Optimization jobs: {status_str}")
        if jobs_complete:
            return
        elapsed += frequency
        if elapsed >= max_wait:
            raise TimeoutError(
                f"Jobs did not complete within {max_wait}s. "
                f"Last status: {status_str}"
            )
        time.sleep(frequency)


def report_errored_records(
    client: PortalClient, record_ids: List[int], logger: logging.Logger,
    section_name: str,
) -> List[int]:
    """List record IDs currently in ERROR state for easy restart.

    Queries the server for every id in ``record_ids``, filters to those
    whose status is ``RecordStatusEnum.error``, and logs them as a single
    space-separated line so the user can copy-paste straight into a reset
    command. No-op (with a brief success message) when no records are
    errored. Returns the list of errored IDs.
    """
    if not record_ids:
        return []
    records = client.get_records(record_ids, missing_ok=True)
    if not isinstance(records, list):
        records = [records]
    errored = [r.id for r in records if r is not None and is_error(r.status)]
    if not errored:
        logger.info(
            f"  All {section_name} records reached terminal state without errors."
        )
        return []
    logger.warning(
        f"\n  {len(errored)} {section_name} record(s) ended in ERROR:"
    )
    logger.warning(f"  {' '.join(str(i) for i in errored)}")
    logger.warning(
        f"  Restart hint: reset these records on the server, then rerun "
        f"the workflow; cached COMPLETE records are picked up via "
        f"find_existing.\n"
    )
    return errored


def check_jobs_status(client: PortalClient, job_ids: List[int],
                      logger: logging.Logger, wait_interval: int = 600,
                      print_job_ids: bool = False,
                      max_wait: int = DEFAULT_MAX_WAIT,
                      auto_recover_services: bool = True) -> None:
    """
    Continuously monitor and report computation status, processing in chunks.

    Parameters
    ----------
    max_wait : int
        Maximum total wait time in seconds (default two weeks).
        Raises ``TimeoutError`` if exceeded.
    auto_recover_services : bool
        If True (default), service records (e.g. ReactionRecord) found at
        ERROR status are reset once per ``check_jobs_status`` call. The
        server re-iterates the service: if its child records are now in
        non-error states (e.g. they were reset externally to recover from
        an infrastructure failure), the service transitions to COMPLETE
        on its own; otherwise it returns to ERROR and stays there. Leaf
        records (singlepoints, optimizations) are never auto-reset.
        Set to False to preserve old behavior (ERROR is fully terminal).
    """
    if not job_ids:
        logger.info("No jobs to monitor.")
        return

    chunk_size = 1000
    elapsed = 0
    reset_attempted: set = set()  # service IDs we've already reset this call

    while True:
        status_counts = {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}
        services_to_recover: List[int] = []

        for i in range(0, len(job_ids), chunk_size):
            chunk = job_ids[i:i + chunk_size]
            if print_job_ids:
                logger.info(f"List with job_ids: {chunk}")
            records = client.get_records(chunk, missing_ok=True)
            if not isinstance(records, list):
                records = [records]

            for job in records:
                if job is not None:
                    label = status_label(job.status)
                    if label in status_counts:
                        status_counts[label] += 1
                    else:
                        logger.info(
                            f"Job ID {job.id}: Unknown status - {job.status}"
                        )

                    if (auto_recover_services
                            and job.status == RecordStatusEnum.error
                            and getattr(job, "is_service", False)
                            and job.id not in reset_attempted):
                        # Conservative recovery: only reset the parent if
                        # NO child is currently in ERROR. A child error
                        # means the service's ERROR is real (data is
                        # missing) and resetting the parent would either
                        # no-op or trigger a server-side cascade that
                        # re-runs the failing child. Leave the parent
                        # alone — the user can reset offending children
                        # externally, and the *next* polling cycle will
                        # see children_errors empty and recover the
                        # parent automatically (matches the 0.15 UX of
                        # "reset child → workflow picks it up, no
                        # restart needed").
                        try:
                            if not job.children_errors:
                                services_to_recover.append(job.id)
                        except Exception:
                            # Transient failure on the children_errors
                            # HTTP call — skip this cycle, try next.
                            pass
                else:
                    logger.info("Job not found in the database")

        logger.info(
            f"Job Status Summary: {status_counts['INCOMPLETE']} INCOMPLETE, "
            f"{status_counts['COMPLETE']} COMPLETE, "
            f"{status_counts['ERROR']} ERROR\n"
        )

        if services_to_recover:
            # Mark IDs as attempted *before* the call so a server-side
            # failure (e.g. transient HTTP 500) doesn't wedge the loop:
            # without this, the same IDs are re-selected every cycle and
            # the workflow never exits. Best-effort: try once, move on.
            reset_attempted.update(services_to_recover)
            logger.info(
                f"Auto-recovering {len(services_to_recover)} errored "
                f"service record(s) — server will re-iterate and transition "
                f"to COMPLETE if children are now OK, else back to ERROR."
            )
            try:
                client.reset_records(services_to_recover)
            except Exception as e:
                logger.warning(
                    f"Auto-recovery reset failed (will not retry these IDs): {e}"
                )

        if status_counts["ERROR"] > 0:
            logger.info("Some jobs have ERROR status. Proceed with caution.")

        if status_counts["INCOMPLETE"] == 0 and not services_to_recover:
            logger.info("All jobs are COMPLETE. Continuing with the execution.")
            return

        elapsed += wait_interval
        if elapsed >= max_wait:
            raise TimeoutError(
                f"Jobs did not complete within {max_wait}s. "
                f"Status: {status_counts['INCOMPLETE']} incomplete, "
                f"{status_counts['COMPLETE']} complete, "
                f"{status_counts['ERROR']} error."
            )
        time.sleep(wait_interval)


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------

def fetch_reaction_entries(client: PortalClient, rdset_base_name: str,
                           stoich: str = "be_nocp"):
    """Get entries and their molecule IDs from a stoich-specific ReactionDataset.

    Returns a list of dicts with keys ``name``, ``stoichiometry``, ``molecule``
    (molecule ID), ``coefficient``, matching the old DataFrame format used by
    ``compute_hessian``.
    """
    import pandas as pd
    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)
    rows = []
    for entry in ds.iterate_entries():
        for s in entry.stoichiometries:
            rows.append({
                "name": entry.name,
                "stoichiometry": stoich,
                "molecule": s.molecule.id,
                "coefficient": s.coefficient,
            })
    return pd.DataFrame(rows)


def fetch_reaction_values(client: PortalClient, rdset_base_name: str,
                          stoich: str = "bsse", spec_name: str = None):
    """Get computed reaction energies from a stoich-specific ReactionDataset.

    Returns a DataFrame with entry names as index and specification names
    as columns, containing ``total_energy`` values converted to kcal/mol.

    Dispersion-corrected DFT is assumed to be stored as a separated pair
    (bare DFT under ``psi4`` + bare dispersion under ``dftd3``/``dftd4``)
    matching the v0.63 migration convention. The bare pieces are summed
    into a composite column (e.g. ``mpwb1k_def2-tzvpd`` + ``mpwb1k-d3bj``
    → ``mpwb1k-d3bj/def2-tzvpd``) for every suffix in
    ``DISPERSION_SUFFIXES``. Integrated specs (method carrying a
    dispersion suffix together with a basis, e.g. ``mpwb1k-d3bj_def2-tzvpd``)
    are skipped with a warning — BEEP no longer supports that layout.

    Column labels are lowercase, matching the lowercase QCSpec convention
    used by every workflow (see ``models/base.py::lowercase_str``).
    """
    import pandas as pd
    import qcelemental as qcel

    logger = logging.getLogger("beep")
    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)

    spec_names = [spec_name] if spec_name else ds.specification_names
    data = {}
    skipped_integrated: List[str] = []
    for sname in spec_names:
        if "_" in sname and _has_dispersion_suffix(sname.split("_", 1)[0]):
            skipped_integrated.append(sname)
            continue

        col_label = sname.replace("_", "/")
        energies = {}
        for entry_name, sn, record in ds.iterate_records(
            specification_names=[sname],
            status=RecordStatusEnum.complete,
        ):
            if record is not None and record.total_energy is not None:
                energies[entry_name] = (
                    record.total_energy * qcel.constants.hartree2kcalmol
                )
        data[col_label] = energies

    if skipped_integrated:
        logger.warning(
            f"  Skipped {len(skipped_integrated)} integrated-dispersion spec(s) "
            f"in {ds_name} (BEEP expects separated bare-DFT + bare-dispersion). "
            f"Re-submit via compute_be_dft_energies to surface these data."
        )

    df = pd.DataFrame(data)

    # Sum bare-DFT + bare-dispersion into composite columns. A bare-dispersion
    # column has no "/" (no basis in its spec name) and ends with one of
    # DISPERSION_SUFFIXES; pair it with a matching "bare/basis" column.
    disp_cols = [
        c for c in df.columns
        if "/" not in c and any(c.endswith(suf) for suf in DISPERSION_SUFFIXES)
    ]
    bare_dft_cols_consumed = set()
    mlp_composite_cols = set()
    for disp_col in disp_cols:
        for suffix in DISPERSION_SUFFIXES:
            if disp_col.endswith(suffix):
                bare = disp_col[: -len(suffix)]
                break
        dft_matches = [c for c in df.columns if c.startswith(bare + "/")]
        if dft_matches:
            for dft_col in dft_matches:
                basis_part = dft_col.split("/", 1)[1]
                composite_col = f"{disp_col}/{basis_part}"
                df[composite_col] = df[dft_col] + df[disp_col]
                bare_dft_cols_consumed.add(dft_col)
        elif bare in df.columns:
            # Range-separated MACE model: the electronic MLP column has no
            # basis (e.g. "h-elec-large"), so there is no "bare/basis" match.
            # Sum it with its dispersion in place — the dispersion column
            # itself becomes the composite MLP+dispersion BE (e.g.
            # "h-elec-large-d4") and the bare electronic column is dropped.
            df[disp_col] = df[bare] + df[disp_col]
            bare_dft_cols_consumed.add(bare)
            mlp_composite_cols.add(disp_col)

    # The bare-DFT and bare-dispersion columns are submission/summing
    # artifacts; user-facing output (log tables, saved JSON) should only
    # carry the composite or integrated columns. Drop them now so every
    # downstream caller sees a clean DataFrame. MLP composite columns
    # (dispersion column reused in place) are kept.
    cols_to_drop = list(bare_dft_cols_consumed) + [
        c for c in disp_cols if c not in mlp_composite_cols
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# ---------------------------------------------------------------------------
# RMSD filter (QCF-coupled version that fetches from server)
# ---------------------------------------------------------------------------

def rmsd_filter_from_dataset(ds_opt, opt_lot: str,
                             logger: logging.Logger) -> Dict[str, Molecule]:
    """
    Filter molecules from an OptimizationDataset by RMSD < 0.25 A.
    Fetches final molecules from the server for comparison.
    """
    logger.info("Starting rmsd_filter")
    molecules_to_delete: List[str] = []
    molecule_records: Dict[str, Molecule] = {}

    for index in ds_opt.entry_names:
        try:
            record = ds_opt.get_record(index, opt_lot)
        except (ValidationError, TypeError) as e:
            logger.warning(
                f"Error retrieving record {index} ({e}); skipping it."
            )
            continue
        if record is None:
            logger.warning(
                f"No record for {index} at {opt_lot}; skipping it."
            )
            continue
        if is_complete(record.status):
            molecule_records[index] = record.final_molecule
        elif is_error(record.status):
            logger.warning(
                f"Optimization of {index} at {opt_lot} finished with ERROR; "
                "it is excluded from the RMSD filter."
            )
        else:
            logger.warning(
                f"Optimization of {index} at {opt_lot} has status "
                f"{status_label(record.status)}; it is excluded from the "
                "RMSD filter."
            )

    molecule_keys: List[str] = list(molecule_records.keys())
    count = 0
    for i in range(len(molecule_keys)):
        for j in range(i + 1, len(molecule_keys)):
            count += 1
            mol1 = molecule_records[molecule_keys[i]]
            mol2 = molecule_records[molecule_keys[j]]
            rmsd = mol1.align(mol2, atoms_map=True)[1]["rmsd"]
            logger.debug(
                f"RMSD between {molecule_keys[i]} and "
                f"{molecule_keys[j]}: {rmsd}"
            )
            # j > i, so a pair is never a self-comparison; an RMSD of
            # exactly 0.0 is a genuine duplicate geometry and must be dropped.
            if rmsd < 0.25:
                if molecule_keys[j] not in molecules_to_delete:
                    molecules_to_delete.append(molecule_keys[j])

    logger.info(f"List of molecules to delete: {molecules_to_delete}")
    unique_molecules_to_delete: List[str] = list(set(molecules_to_delete))
    for molecule_key in unique_molecules_to_delete:
        del molecule_records[molecule_key]

    logger.info(
        f"Remaining molecules after filtering: "
        f"{list(molecule_records.keys())}"
    )
    return molecule_records


# ---------------------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------------------

def create_or_load_reaction_dataset(
    client: PortalClient,
    rdset_name: str,
    opt_lot: str,
    smol_mol: Molecule,
    cluster_mol: Molecule,
    ds_opt,
    opt_stru: Dict[str, object],
    logger: logging.Logger,
    stoich_types: Tuple[str, ...] = STOICH_TYPES,
) -> str:
    """
    Create stoichiometry-specific ReactionDatasets and populate with
    benchmark structures.

    ``stoich_types`` selects which stoichiometry datasets are created and
    populated (default all of :data:`STOICH_TYPES`; pass
    :data:`MACE_STOICH_TYPES` for MLP runs to save only the ghost-free ones).

    Creates one dataset per stoichiometry type (default, be_nocp, ie, de),
    named ``{rdset_name}_{stoich_type}``.

    Returns the base dataset name (without stoich suffix).
    """
    # Get-or-create one dataset per stoich type. Existing datasets are left
    # in place: qcportal 0.63+ ``add_specification`` and ``add_entry`` are
    # idempotent, so a second be_hess run at a different LOT now layers its
    # specs on top of the existing dataset instead of wiping it.
    for stoich_type in stoich_types:
        ds_name = _stoich_dataset_name(rdset_name, stoich_type)
        try:
            client.get_dataset("reaction", ds_name)
        except (KeyError, PortalRequestError):
            client.add_dataset("reaction", ds_name)

    n_entries = 0
    padded_log(
        logger,
        f"Populating the dataframe with {len(opt_stru.keys())} new entries",
        padding_char="*", total_length=60,
    )

    for st in opt_stru.keys():
        logger.info(f"Processing structure: {st}")
        rr = ds_opt.get_record(st, opt_lot)

        # Only COMPLETE optimizations carry a final_molecule; WAITING /
        # RUNNING / ERROR / CANCELLED records would crash be_stoichiometry.
        if rr is None or not is_complete(rr.status):
            status = "missing" if rr is None else status_label(rr.status)
            logger.warning(
                f"WARNING: Optimization of {st} with {opt_lot} is not "
                f"complete (status: {status}). Will skip this structure."
            )
            continue

        struct_mol = rr.final_molecule
        logger.info(f"Generating BE stoichiometry for {st}")
        be_stoich = be_stoichiometry(smol_mol, cluster_mol, struct_mol, logger)
        be_stoich = {k: v for k, v in be_stoich.items() if k in stoich_types}

        try:
            add_reaction(client, rdset_name, st, be_stoich)
        except (KeyError, Exception) as e:
            logger.warning(f"Failed to add {st}. Skipping entry. {e}")
            continue
        n_entries += 1
        logger.info(f"Successfully added {st} to the datasets.\n")

    logger.info(f"Created a total of {n_entries} entries in {rdset_name}.\n")
    return rdset_name


# Dispersion suffixes recognized by BEEP's separated-pair dispersion handling.
# Order matters: longer suffixes first so "-d3mbj" matches before "-d3".
# Single source of truth — consumed by compute_be_dft_energies (write side)
# and fetch_reaction_values / extract.py (read side).
DISPERSION_PROGRAMS: Tuple[Tuple[str, str], ...] = (
    ("-d3mbj", "dftd3"),
    ("-d3bj",  "dftd3"),
    ("-d3m",   "dftd3"),
    ("-d4",    "dftd4"),
    ("-d3",    "dftd3"),
)
DISPERSION_SUFFIXES: Tuple[str, ...] = tuple(s for s, _ in DISPERSION_PROGRAMS)

# Periodic override: QCEngine ships two D3 harnesses. ``dftd3`` is the legacy
# executable wrapper and has NO periodic support -- handed cell/pbc it ignores
# them and silently returns cluster dispersion for a slab. ``s-dftd3`` is the
# python-API harness that honours ``keywords['cell']`` / ``['pbc']``. D4 already
# routes to the python-API ``dftd4``, so it needs no override. Applied only on
# the periodic path: cluster workflows keep ``dftd3`` so their existing
# dispersion specs and records stay valid.
PERIODIC_DISPERSION_PROGRAMS: Dict[str, str] = {"dftd3": "s-dftd3"}


def periodic_dispersion_program(program: Optional[str]) -> Optional[str]:
    """Map a dispersion program name onto its periodic-capable equivalent."""
    if program is None:
        return None
    return PERIODIC_DISPERSION_PROGRAMS.get(program, program)


def _split_dispersion(method: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Split ``method`` into (bare_functional, full_dispersion_method, disp_program).

    Returns (method, None, None) when no known dispersion suffix is present
    (plain DFT, HF-3c, WB97X-V / WB97M-V intrinsic VV10, etc.).
    """
    m = method.lower()
    for suffix, program in DISPERSION_PROGRAMS:
        if m.endswith(suffix):
            bare = method[: -len(suffix)]
            return bare, method, program
    return method, None, None


def _has_dispersion_suffix(name: str) -> bool:
    """Return True if ``name`` ends with a known dispersion suffix."""
    m = name.lower()
    return any(m.endswith(s) for s in DISPERSION_SUFFIXES)


# psi4/dftd3-style dispersion suffix -> (harness keyword field, native keyword
# token) per program. Programs listed here store dispersion-corrected Hessians
# as bare functional + native dispersion keyword; the dict's insertion order is
# also the query order of the get_zpve_mol fallback (keep it deterministic).
# Variants absent for a program (ORCA: -d3m/-d3mbj; Gaussian: -d4/-d3m/-d3mbj)
# raise instead of silently computing with the wrong damping.
DISPERSION_KEYWORD_SPEC: Dict[str, Tuple[str, Dict[str, str]]] = {
    "orca": ("simple_input", {"-d4": "D4", "-d3bj": "D3BJ", "-d3": "D3ZERO"}),
    "gaussian": ("route_input", {"-d3bj": "EmpiricalDispersion=GD3BJ", "-d3": "EmpiricalDispersion=GD3"}),
}


def _keyword_token_match(kw_value: Any, token: str) -> bool:
    """Case-insensitive exact-token match of ``token`` inside a route/simple
    keyword string (whitespace-split, so ``GD3`` never matches ``GD3BJ``)."""
    return token.upper() in str(kw_value).upper().split()


def hessian_method_and_keywords(method: str, mult: int, program: str) -> Tuple[str, dict]:
    """Return the (method, keywords) pair for a Hessian submission on ``program``.

    psi4 (default): the method string is passed through unchanged — psi4
    parses dispersion suffixes itself, so Hessians include the dispersion
    derivative contribution — with psi4-style keywords (``dertype 1``,
    ``reference uks`` for open shells).

    orca / gaussian (any program in ``DISPERSION_KEYWORD_SPEC``): the compound
    method is split into the bare functional plus the program's native
    dispersion keyword on the harness's escape-hatch field (e.g. ``b3lyp-d4``
    -> orca ``simple_input: "D4"``; ``b3lyp-d3bj`` -> gaussian
    ``route_input: "EmpiricalDispersion=GD3BJ"``), because those harnesses
    reject psi4-style compound method strings. The Hessian physics matches
    the psi4 workflow — the same Grimme library supplies the dispersion
    second derivatives. Only the BE-*energy* stage keeps dispersion in
    separate dftd3/dftd4 records. The psi4-style keywords are dropped: both
    programs' Hessians are analytic and UKS/UHF follows from the molecule
    multiplicity.
    """
    spec = DISPERSION_KEYWORD_SPEC.get(program.lower())
    if spec is not None:
        field, table = spec
        bare, disp_method, _ = _split_dispersion(method)
        if disp_method is None:
            return bare, {}
        suffix = disp_method[len(bare):].lower()
        try:
            native_kw = table[suffix]
        except KeyError:
            raise ValueError(
                f"Dispersion variant '{suffix}' of method '{method}' has no native "
                f"{program} keyword ({program} supports {sorted(table)}); "
                "choose a supported variant or run this level of theory with psi4."
            )
        return bare, {field: native_kw}

    kw: dict = {"function_kwargs": {"dertype": 1}}
    if mult != 1:
        kw["reference"] = "uks"
    return method, kw


def compute_be_dft_energies(
    client: PortalClient,
    rdset_base_name: str,
    all_dft: List[str],
    tag: str,
    program: str,
    logger: logging.Logger,
    keyword=None,
) -> List[int]:
    """
    Submit DFT energy computations for BE calculations.

    For dispersion-corrected functionals (``-d3``, ``-d3bj``, ``-d3m``,
    ``-d3mbj``, ``-d4``) two specs are submitted per level of theory: the
    bare DFT piece on ``program`` (typically psi4) and the bare dispersion
    piece on ``dftd3``/``dftd4`` with ``basis=None``. This matches the
    separated-pair form produced by the v0.15→v0.63 migration script and
    allows ``fetch_reaction_values`` to recombine them into composite
    columns. Non-dispersion methods are submitted as a single integrated
    spec. Submits across all stoichiometry-specific datasets. Returns a
    list of record IDs for monitoring.
    """
    logger.info(
        f"Computing energies for the following stoichiometries: "
        f"{' '.join(STOICH_TYPES)} (default = be)\n"
    )

    log_formatted_list(
        logger, all_dft,
        "Sending DFT energy computations for the following Levels of theory:",
        max_rows=1,
    )

    logger.info(f"\nSending DFT computations with tag: {tag}\n")

    all_submitted = 0
    all_existing = 0

    # Local import to avoid a circular module-load dependency between
    # adapters and core/dft_functionals.
    from ..core.dft_functionals import is_3c_method

    spec_names: List[str] = []
    for i, lot in enumerate(all_dft):
        method, basis = split_lot_string(lot)
        bare, disp_method, disp_program = _split_dispersion(method)
        logger.info(f"Processing method: {method}, basis: {basis}")

        lot_submitted = 0
        lot_existing = 0
        # -3c composites already include gCP in the SCF; computing the BSSE
        # counterpoise stoich on top would double-correct.
        skip_stoichs = {"bsse"} if is_3c_method(method) else set()

        for stoich in STOICH_TYPES:
            if stoich in skip_stoichs:
                continue
            if disp_method is None:
                # No dispersion suffix — single integrated spec
                spec_names.append(_energy_spec_name(method, basis))
                result = submit_energies(
                    client, rdset_base_name,
                    method=method, basis=basis, program=program,
                    stoich=stoich, tag=tag, keywords=keyword,
                )
                lot_submitted += result.n_inserted
                lot_existing += result.n_existing
            else:
                # Separated pair: bare DFT + bare dispersion
                spec_names.append(_energy_spec_name(bare, basis))
                spec_names.append(_energy_spec_name(disp_method, None))
                dft_result = submit_energies(
                    client, rdset_base_name,
                    method=bare, basis=basis, program=program,
                    stoich=stoich, tag=tag, keywords=keyword,
                )
                disp_result = submit_energies(
                    client, rdset_base_name,
                    method=disp_method, basis=None, program=disp_program,
                    stoich=stoich, tag=tag, keywords=None,
                )
                lot_submitted += dft_result.n_inserted + disp_result.n_inserted
                lot_existing += dft_result.n_existing + disp_result.n_existing

        all_submitted += lot_submitted
        all_existing += lot_existing
        # 0/0 from ds.submit means the reactions for this LOT were already
        # linked to the dataset from a prior run — neither inserted nor
        # newly-matched in this call. Make that explicit so users don't
        # read a misleading "Existing 0  Submitted 0" as a missing submission.
        if lot_submitted == 0 and lot_existing == 0:
            logger.info(
                f"{lot}: all reactions already linked to the dataset "
                f"(no new submissions)"
            )
        else:
            logger.info(
                f"{lot}: {lot_submitted} newly submitted, "
                f"{lot_existing} newly linked (find_existing)"
            )

    logger.info(
        f"\nSubmitted a total of {all_submitted} DFT computations. "
        f"{all_existing} are newly linked from existing records."
    )

    return _collect_reaction_record_ids(
        client, rdset_base_name, spec_names=spec_names,
    )


def _collect_reaction_record_ids(client: PortalClient,
                                 rdset_base_name: str,
                                 stoich_types: Tuple[str, ...] = STOICH_TYPES,
                                 spec_names: Optional[Sequence[str]] = None) -> List[int]:
    """Collect record IDs across the given stoichiometry datasets for monitoring.

    ``spec_names`` restricts the collection to the specifications submitted
    by the caller. Without it every spec ever registered on the dataset
    (e.g. LOTs of earlier runs) would be monitored, so ``check_jobs_status``
    would wait on and report errors from unrelated levels of theory.
    """
    record_ids = []
    spec_filter = None
    if spec_names is not None:
        spec_filter = sorted({n.lower() for n in spec_names})
    for stoich in stoich_types:
        ds_name = _stoich_dataset_name(rdset_base_name, stoich)
        ds = client.get_dataset("reaction", ds_name)
        for _, _, record in ds.iterate_records(
            specification_names=spec_filter,
            status=[RecordStatusEnum.complete, RecordStatusEnum.running,
                    RecordStatusEnum.waiting, RecordStatusEnum.error],
        ):
            if record is not None:
                record_ids.append(record.id)

    return record_ids


def compute_be_mace_energies(
    client: PortalClient,
    rdset_base_name: str,
    mace_models: List[str],
    tag: str,
    logger: logging.Logger,
    mace_dispersion: Optional[str] = None,
) -> List[int]:
    """
    Submit MACE MLP energy computations for BE calculations.

    Each entry in ``mace_models`` is a path to a serialized MACE model
    file; the spec (and hence dataframe column) is named by the model
    file stem (e.g. ``.../mace-polar-ft0.model`` → ``mace-polar-ft0``).
    Runs through the stock QCEngine MACE harness: ``program='mace'``,
    method = model file path, ``basis=None``.

    Only the ghost-free stoichiometries in :data:`MACE_STOICH_TYPES`
    (``be_nocp``, ``ie_nocp``, ``de``) are computed. The counterpoise
    stoichiometries (``bsse``, ``ie``) are skipped: MLPs carry no basis
    functions so there is no BSSE to correct, and the harness would treat
    ghost atoms as real atoms. The ghost-free interaction energy is
    ``ie_nocp`` (equivalently ``be_nocp - de``).
    """
    log_formatted_list(
        logger, [Path(p).stem for p in mace_models],
        "Sending MACE energy computations for the following models:",
        max_rows=1,
    )
    logger.info(f"\nSending MACE computations with tag: {tag}\n")

    # Range separation: if a dispersion is paired with the (electronic) MLPs,
    # each model additionally gets an analytic dispersion spec (dftd4/s-dftd3,
    # functional params from the prefix) named "<alias><suffix>" so the read
    # side (fetch_reaction_values) sums electronic + dispersion into the
    # composite MLP+dispersion BE, mirroring the DFT-D separated pair.
    # Both spec families share one tag: QCFractal matches tasks to managers on
    # program availability, so a GPU manager without dftd4/s-dftd3 in its env
    # never claims dispersion jobs (and a CPU manager without mace never
    # claims MLP jobs) — no separate routing tag needed.
    disp_program = disp_suffix = None
    if mace_dispersion:
        _bare, _disp_method, disp_program = _split_dispersion(mace_dispersion)
        disp_suffix = mace_dispersion[len(_bare):]
        logger.info(
            f"Range separation: pairing MLPs with {mace_dispersion} "
            f"({disp_program}, spec '<model>{disp_suffix}')\n"
        )

    all_submitted = 0
    all_existing = 0
    spec_names: List[str] = []
    for model_path in mace_models:
        alias = Path(model_path).stem
        logger.info(f"Processing MACE model: {alias} ({model_path})")
        spec_names.append(_energy_spec_name(model_path, None, alias))
        if mace_dispersion:
            spec_names.append(
                _energy_spec_name(mace_dispersion, None, f"{alias}{disp_suffix}")
            )

        model_submitted = 0
        model_existing = 0
        for stoich in MACE_STOICH_TYPES:
            result = submit_energies(
                client, rdset_base_name,
                method=model_path, basis=None, program="mace",
                stoich=stoich, tag=tag, keywords=None,
                spec_name=alias,
            )
            model_submitted += result.n_inserted
            model_existing += result.n_existing

            if mace_dispersion:
                disp_result = submit_energies(
                    client, rdset_base_name,
                    method=mace_dispersion, basis=None, program=disp_program,
                    stoich=stoich, tag=tag, keywords=None,
                    spec_name=f"{alias}{disp_suffix}",
                )
                model_submitted += disp_result.n_inserted
                model_existing += disp_result.n_existing

        all_submitted += model_submitted
        all_existing += model_existing
        if model_submitted == 0 and model_existing == 0:
            logger.info(
                f"{alias}: all reactions already linked to the dataset "
                f"(no new submissions)"
            )
        else:
            logger.info(
                f"{alias}: {model_submitted} newly submitted, "
                f"{model_existing} newly linked (find_existing)"
            )

    logger.info(
        f"\nSubmitted a total of {all_submitted} MACE computations. "
        f"{all_existing} are newly linked from existing records."
    )

    return _collect_reaction_record_ids(
        client, rdset_base_name, MACE_STOICH_TYPES, spec_names=spec_names,
    )


def _hessian_record_matches_dispersion(record, program: str, kw: dict) -> bool:
    """True if an existing Hessian record's dispersion keyword agrees with
    the ``(program, keywords)`` pair :func:`hessian_method_and_keywords`
    would submit.

    orca / gaussian store dispersion in a native keyword on the harness's
    escape-hatch field, so a record at the bare functional matches only if
    that field carries exactly the requested token (or no dispersion token
    when none is requested). psi4 encodes dispersion in the method string,
    so any record carrying a native dispersion token belongs to another
    program's spec and is rejected.
    """
    rec_kw = getattr(getattr(record, "specification", None), "keywords", None) or {}
    if not isinstance(rec_kw, dict):
        return False
    spec = DISPERSION_KEYWORD_SPEC.get(program.lower())
    if spec is not None:
        field, table = spec
        wanted = (kw or {}).get(field)
        rec_val = rec_kw.get(field, "")
        if wanted:
            return _keyword_token_match(rec_val, wanted)
        return not any(_keyword_token_match(rec_val, tok) for tok in table.values())
    for field, table in DISPERSION_KEYWORD_SPEC.values():
        rec_val = rec_kw.get(field, "")
        if any(_keyword_token_match(rec_val, tok) for tok in table.values()):
            return False
    return True


def compute_hessian(
    client: PortalClient,
    ds_be_name: str,
    opt_lot: str,
    mult: int,
    hess_tag: str,
    logger: logging.Logger,
    program: str = "psi4",
) -> List[int]:
    """
    Compute hessian for all molecules in the be_nocp ReactionDataset.
    """
    ds_name = _stoich_dataset_name(ds_be_name, "be_nocp")
    try:
        ds_nocp = client.get_dataset("reaction", ds_name)
    except (KeyError, PortalRequestError):
        logger.info(
            f"\nWARNING: Reaction dataset {ds_name} does not exist.\n"
        )
        return []

    padded_log(
        logger, f"Computing Hessians for {ds_be_name}",
        padding_char="*", total_length=60,
    )

    method, basis = split_lot_string(opt_lot)

    # Collect unique molecule IDs from be_nocp entries
    mol_ids = set()
    for entry in ds_nocp.iterate_entries():
        for s in entry.stoichiometries:
            mol_ids.add(s.molecule.id)

    # Remove single atoms
    u_mols = list(mol_ids)
    mol_list = client.get_molecules(u_mols)
    if not isinstance(mol_list, list):
        mol_list = [mol_list]
    for m in mol_list:
        if len(m.symbols) == 1:
            logger.info(f"\nAtom id {m.id} removed")
            u_mols.remove(m.id)

    log_formatted_list(
        logger, [str(m) for m in u_mols],
        "Sending Hessian computations for the following molecules:",
        max_rows=5,
    )

    method, kw = hessian_method_and_keywords(method, mult, program)

    logger.info(f"\nWill compute Hessian at {method}/{basis} level of theory")
    logger.info(f"\nComputing Hessian at {method}/{basis} level of theory")
    logger.info(f"Using keywords: {kw}")

    # Keyword-agnostic pre-query: match any complete Hessian at the same
    # (mol, method, basis), regardless of the keyword dict that produced
    # it. The DB currently holds a mix of migrated Hessians (stored with
    # empty keywords={}) and newly-submitted ones (stored with the
    # full kw above). add_singlepoints' server-side find_existing matches
    # exactly on the keyword dict, so without this pre-query the migrated
    # records would miss and trigger redundant Hessian re-computes —
    # individually expensive and not what we want for already-done work.
    # The match is keyword-agnostic *except* for the program and the
    # dispersion token: a psi4 Hessian must not satisfy an ORCA/Gaussian
    # request (get_zpve_mol later queries by program for those), and a bare
    # functional query on orca/gaussian must not be satisfied by a record
    # that carries a dispersion keyword (or vice versa).
    existing = [
        r for r in client.query_singlepoints(
            driver=SinglepointDriver.hessian,
            molecule_id=u_mols,
            method=method,
            basis=basis,
            program=program,
            status=RecordStatusEnum.complete,
        )
        if _hessian_record_matches_dispersion(r, program, kw)
    ]
    have = {r.molecule_id for r in existing if r.properties is not None}
    to_submit = [m for m in u_mols if m not in have]
    record_ids = [r.id for r in existing if r.molecule_id in have]

    logger.info(
        f"\nReusing {len(have)} existing Hessian record(s); "
        f"submitting {len(to_submit)} new."
    )

    if to_submit:
        meta, new_ids = client.add_singlepoints(
            molecules=to_submit,
            program=program,
            driver="hessian",
            method=method,
            basis=basis,
            keywords=kw,
            compute_tag=hess_tag,
        )
        record_ids.extend(new_ids)
        logger.info(
            f"\nExisting: {meta.n_existing} Submitted: {meta.n_inserted}"
        )
        logger.info(
            f"\nSubmitted a total of {meta.n_inserted} computations. "
            f"{meta.n_existing} are already computed."
        )
    else:
        logger.info("\nAll Hessians already complete; no new submissions.")
    return record_ids


# ---------------------------------------------------------------------------
# ZPVE (QCF-coupled version)
# ---------------------------------------------------------------------------

def get_zpve_mol(client: PortalClient, mol, lot_opt: str,
                 on_imaginary: str = "return",
                 imag_threshold: float = 50.0):
    """
    Compute vibrational analysis for a molecule and handle imaginary frequencies.
    Fetches hessian data from QCFractal.

    Parameters
    ----------
    imag_threshold : float
        Imaginary frequencies with magnitude below this value (in cm⁻¹)
        are ignored (dropped from ZPVE sum). Default 50 cm⁻¹.
    """
    from ..core.zpve import _vibanal_wfn

    logger = logging.getLogger("beep")
    mol_list = fetch_molecules(client, mol)
    mol_obj = mol_list[0]
    num_atm = len(mol_obj.symbols)

    if num_atm == 1:
        logger.info(
            f"Molecule {mol} is an atom, will return 0.0 for ZPVE"
        )
        return 0.0, True

    mol_form = mol_obj.dict()["identifiers"]["molecular_formula"]
    lot_parts = lot_opt.split("_", 1)
    method = lot_parts[0]
    basis = lot_parts[1] if len(lot_parts) == 2 else None

    results = list(client.query_singlepoints(
        driver=SinglepointDriver.hessian,
        molecule_id=mol,
        method=method,
        basis=basis,
        status=RecordStatusEnum.complete,
    ))

    # orca/gaussian Hessians of dispersion-corrected LOTs are stored under the
    # *bare* functional with the native dispersion keyword on the harness's
    # escape-hatch field (see hessian_method_and_keywords), so the
    # compound-method query above cannot see them. Query the bare method per
    # program (DISPERSION_KEYWORD_SPEC insertion order) and keep only records
    # whose keyword field carries the matching dispersion token.
    bare, disp_method, _ = _split_dispersion(method)
    if disp_method is not None:
        suffix = disp_method[len(bare):].lower()
        for prog, (field, table) in DISPERSION_KEYWORD_SPEC.items():
            native_kw = table.get(suffix)
            if native_kw is None:
                continue
            prog_results = client.query_singlepoints(
                driver=SinglepointDriver.hessian,
                molecule_id=mol,
                method=bare,
                basis=basis,
                program=prog,
                status=RecordStatusEnum.complete,
            )
            results.extend(
                r for r in prog_results
                if _keyword_token_match(r.specification.keywords.get(field, ""), native_kw)
            )

    # Defensive: even a "complete" record can have None properties if the
    # server is in an inconsistent state (e.g. mid-write). Skip those —
    # `result.return_result` dereferences properties and would crash.
    results = [r for r in results if r.properties is not None]
    if not results:
        logger.info(
            f"No complete hessian record at {method}/{basis} for molecule "
            f"{mol} ({mol_form})."
        )
        return None, True
    if len(results) > 1:
        logger.warning(
            f"Found {len(results)} complete hessian records for molecule "
            f"{mol} ({mol_form}) at {method}/{basis}; using the first."
        )
    result = results[0]
    logger.info(
        f"Molecule {mol} with molecular formula {mol_form} "
        "has a Hessian calculation."
    )

    import numpy as np
    hess_raw = result.return_result
    n_coords = 3 * len(result.molecule.symbols)
    hess = np.array(hess_raw).reshape(n_coords, n_coords)
    # Energy may be in extras.qcvars or in properties
    result_dict = result.dict()
    qcvars = result_dict.get("extras", {}).get("qcvars", {})
    energy = qcvars.get("CURRENT ENERGY")
    if energy is None:
        energy = (result.properties or {}).get("current energy")

    result_mol = result.molecule

    vib, therm = _vibanal_wfn(
        hess=hess, molecule=result_mol, energy=energy
    )

    all_imag = [num for num in vib["omega"].data if abs(num.imag) > 1]
    small_imag = [f for f in all_imag if abs(f.imag) < imag_threshold]
    significant_imag = [f for f in all_imag if abs(f.imag) >= imag_threshold]

    if small_imag:
        logger.info(
            f"Molecule {mol}: {len(small_imag)} small imaginary "
            f"frequencies below {imag_threshold} cm⁻¹ ignored: "
            f"{[round(f.imag, 1) for f in small_imag]}"
        )

    if significant_imag:
        if on_imaginary == "raise":
            np.savetxt(f"{mol}_hess.dat", hess, fmt="%.18e")
            raise ValueError(
                f"There are {len(significant_imag)} significant imaginary "
                f"frequencies (>= {imag_threshold} cm⁻¹): {significant_imag}. "
                f"You need to reoptimize {mol}."
            )
        elif on_imaginary == "return":
            if len(significant_imag) > 1:
                logger.warning(
                    f"Molecule {mol}: {len(significant_imag)} significant "
                    f"imaginary frequencies: {significant_imag}. "
                    "Structure may be at a higher-order saddle point."
                )
            else:
                logger.info(
                    f"Molecule {mol}: 1 significant imaginary frequency: "
                    f"{significant_imag[0]}, proceed with caution."
                )
            return therm["ZPE_vib"].data, False
        else:
            raise ValueError(
                f"Invalid option for on_imaginary: {on_imaginary}"
            )

    return therm["ZPE_vib"].data, True


# ---------------------------------------------------------------------------
# Trajectory-benchmark singlepoint helpers (geom_benchmark trajectory mode)
# ---------------------------------------------------------------------------

def get_or_create_singlepoint_dataset(
    client: PortalClient, name: str,
) -> SinglepointDataset:
    """Get-or-create a SinglepointDataset by name."""
    return get_or_create_collection(
        client, name, collection_type=SinglepointDataset,
    )


def add_gradient_spec(
    ds_sp: SinglepointDataset,
    spec_name: str,
    method: str,
    basis: Optional[str],
    program: str = "psi4",
    keywords: Optional[dict] = None,
    description: str = "",
) -> str:
    """Add an SP + gradient ``QCSpecification`` to a ``SinglepointDataset``.

    Idempotent: ``add_specification`` silently reports already-existing
    specs. Returns the lowercased spec name actually registered.
    """
    qc_spec = QCSpecification(
        program=program,
        driver=SinglepointDriver.gradient,
        method=method,
        basis=basis,
        keywords=keywords or {},
    )
    name = spec_name.lower()
    meta = ds_sp.add_specification(
        name=name, specification=qc_spec, description=description,
    )
    _check_insert_meta(meta, f"singlepoint specification '{name}' in {ds_sp.name}")
    return name


def add_energy_spec(
    ds_sp: SinglepointDataset,
    spec_name: str,
    method: str,
    basis: Optional[str],
    program: str = "psi4",
    keywords: Optional[dict] = None,
    description: str = "",
) -> str:
    """Add an SP energy ``QCSpecification`` to a ``SinglepointDataset``.

    Sister of :func:`add_gradient_spec` for the energy driver — used by
    ``be_comp_periodic`` to register the BE electronic + dispersion specs
    on the complex, bare-surface, and gas-phase SinglepointDatasets.
    Idempotent; returns the lowercased spec name actually registered.
    """
    qc_spec = QCSpecification(
        program=program,
        driver=SinglepointDriver.energy,
        method=method,
        basis=basis,
        keywords=keywords or {},
    )
    name = spec_name.lower()
    meta = ds_sp.add_specification(
        name=name, specification=qc_spec, description=description,
    )
    _check_insert_meta(meta, f"singlepoint specification '{name}' in {ds_sp.name}")
    return name


def add_singlepoint_entries(
    ds_sp: SinglepointDataset,
    entries: List[Tuple[str, Molecule]],
) -> Any:
    """Bulk-add ``(entry_name, Molecule)`` pairs to a SinglepointDataset.

    Faster than calling ``add_entry`` in a loop because the server is hit
    once per batch.
    """
    new_entries = [
        SinglepointDatasetNewEntry(name=name, molecule=mol)
        for name, mol in entries
    ]
    meta = ds_sp.add_entries(new_entries)
    _check_insert_meta(meta, f"{len(new_entries)} entries in {ds_sp.name}")
    return meta


def submit_singlepoints_in_dataset(
    ds_sp: SinglepointDataset,
    spec_names: List[str],
    tag: str,
    subset=None,
) -> Any:
    """Submit SP computations from a SinglepointDataset for the given specs."""
    entry_names = list(subset) if subset else None
    return ds_sp.submit(
        entry_names=entry_names,
        specification_names=spec_names,
        compute_tag=tag,
    )


def fetch_sp_energy_gradient(
    ds_sp: SinglepointDataset, entry_name: str, spec_name: str,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Fetch ``(energy_hartree, gradient_hartree_per_bohr)`` from a record.

    Gradient is reshaped from the flat list qcportal returns to
    ``(n_atoms, 3)``. Returns ``(None, None)`` if the record is missing,
    errored, or incomplete.

    Gradient lookup: qcelemental is permissive about where the gradient
    lives — psi4 populates ``properties.return_gradient``; molpro and
    other programs (e.g. via geomeTRIC) only set ``return_result`` on a
    gradient-driver record. We try the properties slot first, then fall
    back to ``return_result`` when the driver is gradient.
    """
    record = ds_sp.get_record(entry_name, spec_name.lower())
    if record is None or not is_complete(record.status):
        return None, None
    props = record.properties or {}
    energy = props.get("return_energy")
    grad = props.get("return_gradient")
    if grad is None and str(record.specification.driver).endswith("gradient"):
        grad = record.return_result
    if grad is not None:
        grad = np.asarray(grad, dtype=float).reshape(-1, 3)
    return energy, grad


def fetch_normal_modes(
    client: PortalClient, mol, hessian_lot: str, return_hessian: bool = False,
):
    """Fetch the complete Hessian record for ``mol`` at ``hessian_lot`` and
    run vibrational analysis. Returns ``(frequencies_cm, modes_cart, mol_obj)``
    where modes are Cartesian, un-mass-weighted, and TR-projected.

    Mirrors the Hessian-record query block in :func:`get_zpve_mol` but
    returns the raw normal-mode data (frequencies + Cartesian eigenvectors)
    instead of the ZPE summary.

    Parameters
    ----------
    client : PortalClient
    mol : int
        Molecule ID on the server.
    hessian_lot : str
        Method + basis as ``method_basis`` (e.g. ``"hf_def2-svp"``).

    Returns
    -------
    frequencies_cm : (n_vib,) complex ndarray, or None if no record
    modes_cart : (n_vib, n_atoms, 3) ndarray, or None if no record
    mol_obj : qcelemental Molecule (the one attached to the record), or None
    hessian : (3N, 3N) ndarray in Eh/bohr², only when ``return_hessian`` is
        True (appended as a fourth element; None if no record)
    """
    from ..core.normal_mode_sampling import extract_normal_modes_from_hessian_record

    logger = logging.getLogger("beep")
    lot_parts = hessian_lot.split("_", 1)
    method = lot_parts[0]
    basis = lot_parts[1] if len(lot_parts) == 2 else None

    results = list(client.query_singlepoints(
        driver=SinglepointDriver.hessian,
        molecule_id=mol,
        method=method,
        basis=basis,
        status=RecordStatusEnum.complete,
    ))
    # Defensive (see get_zpve_mol for backstory): drop records with None
    # properties — accessing return_result on those crashes.
    results = [r for r in results if r.properties is not None]
    if not results:
        logger.info(
            f"No complete hessian record at {method}/{basis} for molecule {mol}."
        )
        return (None, None, None, None) if return_hessian else (None, None, None)
    if len(results) > 1:
        logger.warning(
            f"Found {len(results)} complete hessian records for molecule {mol} "
            f"at {method}/{basis}; using the first."
        )
    record = results[0]
    n_coords = 3 * len(record.molecule.symbols)
    hess = np.asarray(record.return_result, dtype=float).reshape(n_coords, n_coords)

    # Energy is optional — _vibanal_wfn uses it only for the thermo summary
    qcvars = record.dict().get("extras", {}).get("qcvars", {})
    energy = qcvars.get("CURRENT ENERGY")
    if energy is None:
        energy = (record.properties or {}).get("current energy") or 0.0

    freqs_cm, modes_cart = extract_normal_modes_from_hessian_record(
        hess=hess, molecule=record.molecule, energy=energy,
    )
    if return_hessian:
        return freqs_cm, modes_cart, record.molecule, hess
    return freqs_cm, modes_cart, record.molecule


def get_optimization_trajectory(
    odset: OptimizationDataset, entry_name: str, opt_lot: str,
) -> List[dict]:
    """Pull every step of a completed optimization as a list of dicts.

    Each dict has keys ``molecule`` (qcelemental Molecule),
    ``energy_hartree`` (float or None) and ``gradient_hartree_per_bohr``
    (np.ndarray of shape ``(n_atoms, 3)`` or None). Returns ``[]`` if
    the opt is missing, errored, or has an empty trajectory.

    Used by ``geom_benchmark`` trajectory analysis to obtain the shared
    set of geometries (and reference E + ∇E) for cross-functional
    comparison.
    """
    record = odset.get_record(entry_name, opt_lot.lower())
    if record is None or not is_complete(record.status):
        return []
    steps = []
    for sp in record.trajectory or []:
        props = sp.properties or {}
        # Gradient lookup falls back to return_result for programs (e.g.
        # molpro via geomeTRIC) that don't populate properties.return_gradient.
        # See fetch_sp_energy_gradient for the full rationale.
        grad = props.get("return_gradient")
        if grad is None and str(sp.specification.driver).endswith("gradient"):
            grad = sp.return_result
        if grad is not None:
            grad = np.asarray(grad, dtype=float).reshape(-1, 3)
        steps.append({
            "molecule": sp.molecule,
            "energy_hartree": props.get("return_energy"),
            "gradient_hartree_per_bohr": grad,
        })
    return steps


# ---------------------------------------------------------------------------
# MBE / manybody helpers
#
# Support for the ``mbe`` / ``mbe_extract`` workflows (ported from the
# standalone beep-mbe package). These are strictly additive: the monitoring
# functions below carry distinct names (``wait_for_manybody_completion`` /
# ``wait_for_dataset_records``) and do NOT touch the existing
# ``wait_for_completion`` used by sampling / geom_benchmark.
# ---------------------------------------------------------------------------

# Terminal statuses, derived from the real RecordStatusEnum so a record that
# was cancelled / invalidated / deleted server-side stops the polling loop
# instead of being waited on forever (MbeMonitorConfig.max_wait defaults to
# None, i.e. no timeout). "DELETED" is not a RecordStatusEnum member in
# qcportal 0.64 but is kept for forward compatibility.
_MBE_FAILED_STATUSES = frozenset({
    RecordStatusEnum.error.name.upper(),
    RecordStatusEnum.cancelled.name.upper(),
    RecordStatusEnum.invalid.name.upper(),
    "DELETED",
})
_MBE_TERMINAL_STATUSES = frozenset({RecordStatusEnum.complete.name.upper()}) | _MBE_FAILED_STATUSES
_MBE_CHILD_STATUS_KEYS = ("WAITING", "RUNNING", "COMPLETE", "ERROR")


@dataclass
class ManybodyMonitorResult:
    """Structured result of monitoring an MBE ManybodyDataset submission."""
    start_time: float
    end_time: float
    polls: int
    per_entry_final_status: Dict[str, str]
    per_entry_children_counts: Dict[str, Dict[str, int]]
    n_complete: int
    n_error: int
    n_missing: int
    n_other: int
    errored_entries: List[str]
    timed_out: bool


def get_or_create_manybody_dataset(client: PortalClient, name: str) -> ManybodyDataset:
    """Get an existing ManybodyDataset or create a new one (idempotent)."""
    try:
        return client.get_dataset("manybody", name)
    except (KeyError, PortalRequestError):
        return client.add_dataset("manybody", name)


def list_reaction_dataset_names(client: PortalClient) -> List[str]:
    """Read-only list of all ReactionDataset names on the server.

    Used by the MBE ZPVE-borrow auto-discovery to locate ``be_<MOL>_*`` datasets.
    """
    return [
        r["dataset_name"]
        for r in client.list_datasets()
        if r.get("dataset_type") == "reaction"
    ]


def mbe_levels_to_qc_specifications(levels, program: str) -> Dict[int, QCSpecification]:
    """Convert MBE level objects to per-order energy QCSpecifications.

    ``levels`` is any iterable of objects exposing ``.index``, ``.method``,
    ``.basis`` and ``.keywords`` (e.g. :class:`beep.models.mbe.MbeLevel`).
    Returns a mapping from MBE order index to a QCSpecification.
    """
    qc_levels: Dict[int, QCSpecification] = {}
    for level in levels:
        qc_levels[level.index] = QCSpecification(
            program=program,
            driver=SinglepointDriver.energy,
            method=level.method,
            basis=level.basis,
            # QCSpecification rejects keywords=None; use an empty dict when the
            # level carries no keywords.
            keywords=level.keywords or {},
        )
    return qc_levels


def build_manybody_specification(
    levels: Dict[int, QCSpecification], bsse_correction: List[str],
) -> ManybodySpecification:
    """Build a qcmanybody ManybodySpecification from per-order QCSpecifications."""
    return ManybodySpecification(
        program="qcmanybody",
        levels=levels,
        bsse_correction=bsse_correction,
        keywords=ManybodyKeywords(return_total_data=True),
    )


def _mbe_normalize_status(value) -> str:
    if value is None:
        return "MISSING"
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    return str(value).upper()


def _mbe_normalize_children_status(children_status) -> Dict[str, int]:
    counts: Dict[str, int] = {key: 0 for key in _MBE_CHILD_STATUS_KEYS}
    if not children_status:
        return counts
    for key, value in children_status.items():
        status_key = _mbe_normalize_status(key)
        if status_key in counts:
            counts[status_key] += int(value)
    return counts


def _mbe_log_poll_summary(poll_index, entries, statuses, children_counts) -> None:
    logger = logging.getLogger("beep")
    logger.info(f"Monitoring poll {poll_index}: {len(entries)} entries")
    for entry in entries:
        status = statuses[entry]
        counts = children_counts[entry]
        total = sum(counts.values())
        logger.info(
            f"entry={entry} status={status} children_total={total} "
            f"waiting={counts['WAITING']} running={counts['RUNNING']} "
            f"complete={counts['COMPLETE']} error={counts['ERROR']}"
        )


def _mbe_summarize_final_statuses(per_entry_final_status):
    logger = logging.getLogger("beep")
    n_complete = n_error = n_missing = n_other = 0
    errored_entries = []
    for entry, status in per_entry_final_status.items():
        if status == "COMPLETE":
            n_complete += 1
        elif status in _MBE_FAILED_STATUSES:
            n_error += 1
            errored_entries.append(entry)
            if status != "ERROR":
                logger.warning(
                    f"Entry '{entry}' reached terminal failure status {status} "
                    "(cancelled/invalidated/deleted on the server); it will "
                    "not complete without manual intervention."
                )
        elif status == "MISSING":
            n_missing += 1
        else:
            n_other += 1
    return n_complete, n_error, n_missing, n_other, errored_entries


def wait_for_manybody_completion(
    client: PortalClient,
    dataset_name: str,
    spec_name: str,
    entry_names: Sequence[str],
    poll_interval_s: int = 300,
    max_wait_s: Optional[int] = None,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.time,
) -> ManybodyMonitorResult:
    """Poll ManybodyDataset record statuses until all entries reach terminal states.

    Distinct from :func:`wait_for_completion` (which monitors a flat list of
    record IDs for the reaction-dataset BE route). ``sleep_fn`` / ``time_fn``
    are injectable to keep the polling loop deterministic under test.
    """
    logger = logging.getLogger("beep")
    start_time = time_fn()
    polls = 0
    timed_out = False

    mb_ds = client.get_dataset("manybody", dataset_name)
    try:
        overview = mb_ds.detailed_status()
    except Exception as exc:
        logger.debug(f"Dataset detailed_status unavailable: {exc}")
    else:
        logger.info(f"Dataset detailed status snapshot: {overview}")

    per_entry_status: Dict[str, str] = {}
    per_entry_children: Dict[str, Dict[str, int]] = {}

    while True:
        polls += 1
        for entry in entry_names:
            rr = mb_ds.get_record(entry_name=entry, specification_name=spec_name)
            if rr is None:
                per_entry_status[entry] = "MISSING"
                per_entry_children[entry] = {key: 0 for key in _MBE_CHILD_STATUS_KEYS}
                continue
            per_entry_status[entry] = _mbe_normalize_status(rr.status)
            per_entry_children[entry] = _mbe_normalize_children_status(rr.children_status)

        _mbe_log_poll_summary(polls, entry_names, per_entry_status, per_entry_children)

        if all(s in _MBE_TERMINAL_STATUSES for s in per_entry_status.values()):
            break

        elapsed = time_fn() - start_time
        if max_wait_s is not None and elapsed >= max_wait_s:
            timed_out = True
            logger.warning(
                f"Monitoring timed out after {elapsed:.1f} seconds (max_wait={max_wait_s})."
            )
            break

        sleep_fn(poll_interval_s)

    end_time = time_fn()
    n_complete, n_error, n_missing, n_other, errored_entries = _mbe_summarize_final_statuses(
        per_entry_status
    )
    logger.info(
        f"Monitoring complete: complete={n_complete} error={n_error} "
        f"missing={n_missing} other={n_other}"
    )
    if errored_entries:
        logger.info(f"Errored entries: {', '.join(errored_entries)}")

    return ManybodyMonitorResult(
        start_time=start_time,
        end_time=end_time,
        polls=polls,
        per_entry_final_status=dict(per_entry_status),
        per_entry_children_counts=dict(per_entry_children),
        n_complete=n_complete,
        n_error=n_error,
        n_missing=n_missing,
        n_other=n_other,
        errored_entries=list(errored_entries),
        timed_out=timed_out,
    )


def wait_for_dataset_records(
    ds,
    entry_names: Sequence[str],
    specification_names: Sequence[str],
    poll_interval: int,
    max_wait: Optional[int],
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.time,
) -> Tuple[Dict[Tuple[str, str], str], bool]:
    """Poll a dataset's entry/spec record statuses until all are terminal.

    Used for the monomer SinglepointDataset in the MBE workflow. Returns the
    per-(entry, spec) status map and a timed-out flag.
    """
    logger = logging.getLogger("beep")
    start_time = time_fn()
    polls = 0
    timed_out = False
    per_record_status: Dict[Tuple[str, str], str] = {}

    while True:
        polls += 1
        for entry in entry_names:
            for spec in specification_names:
                rr = ds.get_record(entry_name=entry, specification_name=spec)
                per_record_status[(entry, spec)] = _mbe_normalize_status(
                    rr.status if rr else None
                )

        counts = Counter(per_record_status.values())
        known = (
            counts.get("COMPLETE", 0) + counts.get("ERROR", 0)
            + counts.get("WAITING", 0) + counts.get("RUNNING", 0)
            + counts.get("MISSING", 0)
        )
        logger.info(
            f"Dataset poll {polls}: total={len(per_record_status)} "
            f"complete={counts.get('COMPLETE', 0)} error={counts.get('ERROR', 0)} "
            f"waiting={counts.get('WAITING', 0)} running={counts.get('RUNNING', 0)} "
            f"missing={counts.get('MISSING', 0)} other={sum(counts.values()) - known}"
        )

        if all(s in _MBE_TERMINAL_STATUSES for s in per_record_status.values()):
            for (entry, spec), s in per_record_status.items():
                if s in _MBE_FAILED_STATUSES and s != "ERROR":
                    logger.warning(
                        f"Record {entry}/{spec} reached terminal failure status "
                        f"{s} (cancelled/invalidated/deleted on the server)."
                    )
            break

        elapsed = time_fn() - start_time
        if max_wait is not None and elapsed >= max_wait:
            timed_out = True
            logger.warning(
                f"Dataset monitoring timed out after {elapsed:.1f} seconds (max_wait={max_wait})."
            )
            break

        sleep_fn(poll_interval)

    logger.info(f"Dataset monitoring complete: total={len(per_record_status)}")
    return dict(per_record_status), timed_out


def add_reaction_energy_spec(
    ds_rxn: ReactionDataset,
    spec_name: str,
    method: str,
    basis: Optional[str],
    program: str = "psi4",
    keywords: Optional[dict] = None,
    description: str = "",
) -> str:
    """Add an energy ``ReactionSpecification`` to a ``ReactionDataset``.

    The singlepoint part is built exactly like :func:`add_energy_spec`, so the component
    records of a reaction are the same records a SinglepointDataset with that spec would hold
    (QCFractal deduplicates them). Idempotent; returns the lowercased spec name.
    """
    qc_spec = QCSpecification(
        program=program,
        driver=SinglepointDriver.energy,
        method=method,
        basis=basis,
        keywords=keywords or {},
    )
    name = spec_name.lower()
    rxn_spec = ReactionSpecification(program="reaction", singlepoint_specification=qc_spec,
                                     keywords=ReactionKeywords())
    meta = ds_rxn.add_specification(name=name, specification=rxn_spec, description=description)
    _check_insert_meta(meta, f"reaction specification '{name}' in {ds_rxn.name}")
    return name


def reaction_component_energies(record) -> Optional[List[Tuple[float, int, float]]]:
    """``[(coefficient, n_atoms, energy)]`` of a complete reaction record's components, or None
    when the record or any component is missing or not complete. Fetch the records with
    ``include=["components"]`` so this needs no further server calls."""
    if record is None or not is_complete(record.status):
        return None
    out = []
    for comp in record.components:
        sp = comp.singlepoint_record
        if sp is None or not is_complete(sp.status):
            return None
        e = (sp.properties or {}).get("return_energy")
        if e is None:
            return None
        out.append((float(comp.coefficient), len(comp.molecule.symbols), float(e)))
    return out
