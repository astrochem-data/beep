"""Normal-mode displacement sampling workflow config."""
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from .base import ServerConfig, lowercase_str, QC_KEYWORDS_DESCRIPTION, deprecated_null_only


class BandSpec(BaseModel):
    """Per-band cap on the number of modes to pick + RMS Cartesian
    displacement amplitude (Å) for each pick."""
    cap: int = Field(
        ..., ge=0,
        description="Maximum number of modes to pick from this band per system",
    )
    amplitude_A: float = Field(
        ..., gt=0.0,
        description=(
            "RMS Cartesian displacement (Å) of all atoms for a ± pair. "
            "Each picked mode contributes 2 displaced structures at "
            "amplitudes (+amp, -amp)."
        ),
    )


def _default_bands() -> Dict[str, BandSpec]:
    """Defaults tuned for H₂O/H₂S cluster + small-molecule binding sites.

    Total per system: (3 + 2 + 1) × 2 = 12 displaced structures, +2 if the
    lowest-frequency mode gets a second amplitude → 14 / system → ~70
    CCSD(T) gradients on a 5-site benchmark.
    """
    return {
        "intermolecular": BandSpec(cap=3, amplitude_A=0.08),
        "bending":        BandSpec(cap=2, amplitude_A=0.05),
        "stretching":     BandSpec(cap=1, amplitude_A=0.03),
    }


class NmSamplingConfig(BaseModel):
    """Configuration for the normal-mode displacement benchmark workflow.

    Works on any set of optimised geometries stored in a single
    ``OptimizationDataset`` — adsorbate-on-cluster, N-mer clusters,
    or any partition into rigid-body fragments. Per system:

      1. Pulls the equilibrium geometry from ``opt_dataset`` at
         ``geometry_opt_lot``.
      2. Computes a Hessian at ``hessian_lot`` (default ``hf_def2-svp``).
      3. Diagonalises it (via qcelemental ``vibanal``), classifies each
         normal mode as intermolecular / bending / stretching using
         fragment-COM + rotation-about-COM projection summed over the
         fragments given by ``fragments``.
      4. Picks the lowest-frequency modes from each band up to its cap;
         the N lowest-frequency selected modes get a second amplitude.
      5. Generates ± displaced geometries at the per-band amplitude.
      6. Submits a CCSD(T) gradient on each displaced geometry as the
         reference, plus a DFT gradient per functional in the same
         curated functional list ``geom_benchmark`` uses.
      7. Reports per-functional force-RMSD vs the CCSD(T) reference,
         per-category, with the same log layout as the trajectory
         benchmark in ``geom_benchmark``.
    """
    workflow: Literal["nm_sampling"] = Field(..., description="Must be 'nm_sampling'")
    server: ServerConfig = Field(default_factory=ServerConfig, description="QCFractal server connection settings")
    opt_dataset: str = Field(
        ...,
        description=(
            "Name of the OptimizationDataset containing every entry in "
            "``benchmark_structures``. Entry names are free-form labels; "
            "no rsplit / dataset-name-in-entry-name convention required."
        ),
    )
    benchmark_structures: List[str] = Field(
        ...,
        description=(
            "Entry names within ``opt_dataset`` to benchmark. "
            "All entries must have the same atom-count layout matching ``fragments``."
        ),
    )
    fragments: Dict[str, List[List[int]]] = Field(
        ...,
        description=(
            "Per-benchmark-structure fragment partition. Keys must cover every "
            "entry in ``benchmark_structures`` exactly; each value is a list of "
            "0-indexed atom-index lists matching the shape of "
            "``qcel.Molecule.fragments``. Every atom must appear in exactly one "
            "fragment. Use one fragment per monomer — collapsing multiple "
            "monomers into a single rigid block will label intra-block "
            "rearrangements as bending/stretching (wrong), because they are "
            "not rigid-body motion of the collapsed block. Example (water "
            "trimer, 9 atoms, three monomers): "
            "``{\"h2o_3\": [[0,1,2], [3,4,5], [6,7,8]]}``."
        ),
    )
    geometry_opt_lot: str = Field(
        ...,
        description=(
            "Level of theory the optimised binding-site geometries live at "
            "(method_basis, e.g. 'mpwb1k-d3bj_def2-tzvpd'). The Hessian and "
            "all displacement gradients are evaluated at THIS geometry."
        ),
    )

    # --- Hessian ---
    hessian_lot: str = Field(
        "hf_def2-svp",
        description=(
            "Level of theory for the Hessian. Default 'hf_def2-svp' — the "
            "Hessian is only used as a basis for displacement directions, "
            "so a cheap SCF is sufficient. Configurable for cases where a "
            "richer normal-mode picture is wanted (e.g. 'pbe-d4_def2-svp')."
        ),
    )
    hessian_program: str = Field("psi4", description="QC program for the Hessian")
    hessian_keywords: Optional[Dict[str, str]] = Field(
        None,
        description="QC keywords passed to the Hessian SP submission (e.g. {'scf_type': 'df'})",
    )
    tag_hessian: Optional[str] = Field(None, description="Queue tag for the Hessian computations")

    # --- Reference CCSD(T) gradients ---
    reference_grad_lot: str = Field(
        "ccsd(t)_aug-cc-pvtz",
        description="Reference level of theory for the per-displacement gradients (method_basis)",
    )
    reference_grad_program: str = Field("psi4", description="QC program for the reference gradient")
    reference_grad_keywords: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {"scf_type": "df", "cc_type": "df", "freeze_core": "true"},
        description="QC keywords for the reference gradient submissions",
    )
    tag_reference_grad: Optional[str] = Field(None, description="Queue tag for the reference (CCSD(T)) gradients")

    # --- DFT gradients ---
    dft_program: str = Field("psi4", description="QC program for the DFT gradients")
    qc_keywords: Optional[Dict[str, Any]] = Field(
        None,
        description=QC_KEYWORDS_DESCRIPTION + " Applied to every DFT gradient spec.",
    )
    dft_keyword: SkipJsonSchema[Optional[Any]] = Field(
        None, description="Deprecated, use qc_keywords.", exclude=True,
    )
    _dep_dft_keyword = deprecated_null_only("dft_keyword", "qc_keywords")
    tag_dft_grad: Optional[str] = Field(None, description="Queue tag for the DFT gradient computations")

    # --- Mode selection ---
    bands: Dict[str, BandSpec] = Field(
        default_factory=_default_bands,
        description=(
            "Per-band cap + amplitude. Recognised band names are "
            "'intermolecular', 'bending', 'stretching' (matching the "
            "labels emitted by classify_mode). Defaults give 6 modes × "
            "2 (± per mode) = 12 displaced structures per system."
        ),
    )
    inter_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description=(
            "Fragment-COM-projection ratio above which a mode is labelled "
            "intermolecular. f_inter = inter-fragment kinetic / total "
            "kinetic ∈ [0, 1]."
        ),
    )
    bend_max_cm: float = Field(
        1500.0, gt=0.0,
        description=(
            "Intramolecular modes (f_inter ≤ inter_threshold) with a "
            "frequency below this cutoff are labelled 'bending'; above "
            "this cutoff, 'stretching'."
        ),
    )
    freq_max_imag_cm: float = Field(
        50.0, ge=0.0,
        description=(
            "Maximum allowed magnitude (cm⁻¹) of an imaginary frequency "
            "for a mode to be eligible for displacement. Modes with "
            "|imag| above this are dropped (genuine saddle-point directions)."
        ),
    )
    extra_amplitudes_lowest_count: int = Field(
        1, ge=0,
        description=(
            "Number of lowest-frequency selected modes that get a second "
            "amplitude. Each extra amplitude adds two displaced structures "
            "(+, −) at amplitude × extra_amplitude_factor."
        ),
    )
    extra_amplitude_factor: float = Field(
        2.0, gt=0.0,
        description="Multiplier on the band amplitude for the extra-amplitude entries",
    )

    # --- Reporting ---
    geometry_shift_min_freq_cm: float = Field(
        20.0, ge=0.0,
        description=(
            "Normal modes below this frequency (cm⁻¹) are excluded when the "
            "force offset is converted into a predicted geometry shift "
            "(Δx = H⁺·ΔF): for near-free motions 1/k amplifies noise."
        ),
    )
    ranking_metric: Literal["total", "response", "intermolecular"] = Field(
        "intermolecular",
        description=(
            "Force metric that sorts the decomposition table and picks the "
            "best functional per group. "
            "'intermolecular' (default) = RMSD of the net force per fragment, "
            "the part relevant to binding. "
            "'total' = RMSD over every Cartesian force component (the pre-"
            "2026-10 behaviour; "
            "dominated by stiff bond stretches, so a functional whose bond "
            "lengths differ slightly from the reference scores badly even "
            "at intermolecular displacements). 'response' = RMSD of "
            "(F+ − F−)/2 over ± pairs: the force error that grows with the "
            "displacement, with the error already present at the "
            "displacement centre cancelled. All metrics are always reported."
        ),
    )

    # --- Run-mode controls ---
    pre_run: bool = Field(
        False,
        description=(
            "If true, run only the Hessian + normal-mode + classification "
            "stages and exit before submitting any gradient SPs. The .molden "
            "files and per-system mode JSONs are written so the user can "
            "visually inspect the modes before committing the gradient budget. "
            "Re-run with pre_run=false to continue (Hessians dedup, no recompute)."
        ),
    )
    allow_imaginary_modes: bool = Field(
        False,
        description=(
            "By default the workflow aborts if any system has an imaginary "
            "frequency with magnitude above freq_max_imag_cm (saddle "
            "geometry — displacements there would be meaningless). Set to "
            "true to proceed anyway (e.g. for transition-state probing)."
        ),
    )

    # --- Lowercase validators (qcportal stores spec names lowercase) ---
    _lower_geom_opt = field_validator("geometry_opt_lot")(lowercase_str)
    _lower_hess_lot = field_validator("hessian_lot")(lowercase_str)
    _lower_ref_lot = field_validator("reference_grad_lot")(lowercase_str)

    @model_validator(mode="after")
    def _validate_fragments(self):
        if not self.fragments:
            raise ValueError("fragments must be non-empty.")
        expected = set(self.benchmark_structures)
        missing = expected - set(self.fragments.keys())
        extra = set(self.fragments.keys()) - expected
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
        for struct_name, partition in self.fragments.items():
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
        return self
