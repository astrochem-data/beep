"""Base config models shared across workflows."""
import json
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


QC_KEYWORDS_DESCRIPTION = (
    "Extra QC-program keywords (psi4/ORCA/Gaussian options such as "
    "{'guess': 'gwh', 'damping_percentage': 20}) passed as the "
    "specification's qc_spec keywords. Distinct from *_opt_keywords, "
    "which are geomeTRIC optimizer options."
)


FRAGMENTS_DESCRIPTION = (
    "Per-benchmark-structure fragment partition. Keys must cover every "
    "entry in ``benchmark_structures`` exactly; each value is a list of "
    "0-indexed atom-index lists matching the shape of "
    "``qcel.Molecule.fragments``. Every atom must appear in exactly one "
    "fragment. Use one fragment per monomer — collapsing multiple "
    "monomers into a single rigid block mislabels intra-block "
    "rearrangements as intramolecular. Example (water trimer, 9 atoms, "
    "three monomers): ``{\"h2o_3\": [[0,1,2], [3,4,5], [6,7,8]]}``."
)


def validate_fragment_partition(fragments, benchmark_structures):
    """Check a ``{structure → [[atom idx, ...], ...]}`` partition.

    Keys must match ``benchmark_structures`` exactly; every fragment must
    be non-empty, indices non-negative and disjoint across fragments.
    The atom count itself can only be checked against the stored
    molecule at run time.
    """
    if not fragments:
        raise ValueError("fragments must be non-empty.")
    expected = set(benchmark_structures)
    missing = expected - set(fragments.keys())
    extra = set(fragments.keys()) - expected
    if missing:
        raise ValueError(
            f"fragments is missing entries for benchmark structure(s): "
            f"{sorted(missing)}."
        )
    if extra:
        raise ValueError(
            f"fragments has entries for structure(s) not in "
            f"benchmark_structures: {sorted(extra)}."
        )
    for struct_name, partition in fragments.items():
        if not partition:
            raise ValueError(f"fragments[{struct_name!r}] is empty.")
        seen: set = set()
        for i, frag in enumerate(partition):
            if not frag:
                raise ValueError(
                    f"fragments[{struct_name!r}][{i}] is empty."
                )
            for a in frag:
                if a < 0:
                    raise ValueError(
                        f"fragments[{struct_name!r}][{i}] contains negative "
                        f"atom index {a}."
                    )
                if a in seen:
                    raise ValueError(
                        f"atom index {a} appears in more than one fragment "
                        f"for {struct_name!r} — fragments must be disjoint."
                    )
                seen.add(a)
    return fragments


def deprecated_null_only(old: str, new: str):
    """Validator for a removed config field: null passes, anything else errors pointing to `new`.

    Use as a class attribute on the model, next to the deprecated field::

        keyword_id: SkipJsonSchema[Optional[Any]] = Field(
            None, description="Deprecated, use qc_keywords.", exclude=True)
        _dep_keyword_id = deprecated_null_only("keyword_id", "qc_keywords")

    The field stays declared so old configs that carry ``"<old>": null``
    still load; ``exclude=True`` keeps it out of ``model_dump`` (and thus
    the config copy every workflow writes) and ``SkipJsonSchema`` out of
    ``beep --schema``.
    """
    def _reject_non_null(v):
        if v is not None:
            raise ValueError(
                f"`{old}` is no longer supported (QCFractal >= 0.50 stores program "
                f"keywords inline, not by server-side ID); put the options in "
                f"`{new}` as a dict"
            )
        return v
    return field_validator(old)(_reject_non_null)


def safe_config_dump(config) -> str:
    """JSON-serialize a workflow config without exposing credentials.

    Strips ``server.password`` and ``server.username`` before dumping.
    Every workflow writes a copy of its input config to disk for
    reproducibility; this helper ensures that copy never contains
    plaintext credentials.
    """
    return json.dumps(
        config.model_dump(exclude={"server": {"password", "username"}}),
        indent=4,
        default=str,
    )


def lowercase_str(v):
    """Normalize a method/basis string to lowercase. Skips None and empty.

    BEEP standardised on lowercase QCSpec names after the qcportal 0.63
    migration; the server stores specs lowercase and case-sensitive lookups
    fail otherwise.
    """
    if v is None or not isinstance(v, str) or not v:
        return v
    return v.lower()


def lowercase_list(v):
    """Normalize a list of method/basis strings to lowercase (per-item)."""
    if v is None:
        return v
    return [s.lower() if isinstance(s, str) and s else s for s in v]


class ServerConfig(BaseModel):
    """QCFractal server connection settings."""
    address: str = Field("localhost:7777", description="Server address (host:port)")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    # Default is `False` because BEEP deployments typically target QCFractal
    # servers on internal HTTP-only addresses (e.g. `http://qcf-astrochem`) or
    # reverse-proxied with TLS terminated upstream — there is no client-side
    # certificate to verify. Set to `True` when the server is reachable over
    # HTTPS with a publicly-trusted certificate.
    verify: bool = Field(
        False,
        description=(
            "Verify the server's TLS certificate. Defaults to False for "
            "internal HTTP-only QCFractal deployments; set True for "
            "HTTPS servers with a trusted certificate."
        ),
    )


def split_lot_string(lot: str) -> Tuple[str, Optional[str]]:
    """Split a ``method_basis`` LOT string into ``(method, basis)``.

    Basis-less LOTs (``gfn2-xtb``, MACE model aliases) return ``basis=None``.
    Only the first underscore separates method from basis, matching the
    spec-name convention used across all workflows.
    """
    parts = lot.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def validate_mace_model_path(v):
    """Validate a MACE model file path used as a QCSpec method.

    qcportal force-lowercases QCSpec method strings server-side, so the
    path must already be all-lowercase or record lookups would silently
    miss. The file stem becomes the spec/column/dataset alias, and BEEP
    splits LOT strings on ``_``, so the stem must not contain underscores.
    """
    if v is None:
        return v
    if v != v.lower():
        raise ValueError(
            f"mace_model path must be all-lowercase (qcportal lowercases "
            f"QCSpec methods server-side): '{v}'"
        )
    stem = Path(v).stem
    if "_" in stem:
        raise ValueError(
            f"mace_model file stem must not contain underscores (BEEP "
            f"splits LOT names on '_'); use hyphens instead: '{stem}'"
        )
    return v


class LevelOfTheory(BaseModel):
    """A quantum chemistry level of theory.

    Either a conventional QC method (``method`` + optional ``basis`` on
    ``program``) or a MACE machine-learning potential via ``mace_model``.
    When ``mace_model`` is set, ``method``/``basis``/``program`` are muted:
    the spec runs as ``program='mace'`` with the model file path as method
    (stock QCEngine MACE harness), and the model file stem (e.g.
    ``mace-polar-ft0`` for ``.../mace-polar-ft0.model``) is used as the
    LOT name for specs, datasets, columns, and files.
    """
    method: Optional[str] = Field(None, description="QC method name (e.g. 'hf', 'b3lyp-d3bj'); required unless mace_model is set")
    basis: Optional[str] = Field(None, description="Basis set name (e.g. 'def2-svp')")
    program: str = Field("psi4", description="QC program to use")
    mace_model: Optional[str] = Field(
        None,
        description=(
            "Path to a serialized MACE model file (all-lowercase, stem "
            "without underscores). Mutes method/basis/program. Supported "
            "by the sampling, be_hess, and extract workflows."
        ),
    )

    _lower_method = field_validator("method")(lowercase_str)
    _lower_basis = field_validator("basis")(lowercase_str)
    _check_mace_model = field_validator("mace_model")(validate_mace_model_path)

    @model_validator(mode="after")
    def _require_method_or_mace(self):
        if self.mace_model is None and self.method is None:
            raise ValueError("LevelOfTheory requires either 'method' or 'mace_model'")
        return self

    @property
    def is_mace(self) -> bool:
        return self.mace_model is not None

    @property
    def alias(self) -> Optional[str]:
        """LOT name for a MACE model: the model file stem."""
        if self.mace_model is None:
            return None
        return Path(self.mace_model).stem

    @property
    def lot_name(self) -> str:
        """Name used for specs/datasets/columns: alias or method[_basis]."""
        if self.is_mace:
            return self.alias
        if self.basis:
            return f"{self.method}_{self.basis}".lower()
        return self.method.lower()

    @property
    def qc_method(self) -> str:
        """Method string for the QCSpecification (model path for MACE)."""
        return self.mace_model if self.is_mace else self.method

    @property
    def qc_basis(self) -> Optional[str]:
        return None if self.is_mace else self.basis

    @property
    def qc_program(self) -> str:
        return "mace" if self.is_mace else self.program

    @property
    def display(self) -> str:
        """Human-readable LOT for logs and config summaries."""
        if self.is_mace:
            return f"{self.alias} (mace)"
        return f"{self.method}/{self.basis or 'N/A'} ({self.program})"
