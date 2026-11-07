"""Geometry benchmark workflow config — entry-based, single OptimizationDataset."""
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from .base import (
    ServerConfig, QC_KEYWORDS_DESCRIPTION, FRAGMENTS_DESCRIPTION,
    deprecated_null_only, validate_fragment_partition,
)


class GeomBenchmarkConfig(BaseModel):
    """Configuration for the geometry benchmark workflow.

    Entry-based: every geometry to benchmark is an entry in a single
    ``opt_dataset`` (OptimizationDataset). Each listed entry is optimized
    at the reference LOT and at every DFT functional in the curated
    lists, then compared by RMSD (and optionally per-step force RMSD
    along the reference trajectory). There is no adsorbate/surface
    special-casing — to benchmark a monomer or bare surface alongside
    the complexes, add it as another entry in ``opt_dataset``.
    """
    workflow: Literal["geom_benchmark"] = Field(..., description="Must be 'geom_benchmark'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    opt_dataset: str = Field(
        ...,
        description=(
            "Name of the OptimizationDataset containing every entry in "
            "``benchmark_structures``. Entry names are free-form labels."
        ),
    )
    benchmark_structures: List[str] = Field(
        ...,
        description=(
            "Entry names within ``opt_dataset`` to benchmark. Each stored "
            "molecule carries its own charge/multiplicity — no global "
            "adsorbate multiplicity is assumed."
        ),
    )
    reference_geometry_level_of_theory: List[str] = Field(
        ["ccsd(t)", "aug-cc-pvtz", "psi4"],
        description="Reference geometry level of theory [method, basis, program]",
    )
    reference_geometry_keywords: Optional[Dict[str, str]] = Field(
        {"scf_type": "df", "cc_type": "df", "freeze_core": "true"},
        description="QC keywords for reference geometry (e.g. scf_type, cc_type)",
    )
    tag_reference_geometry: Optional[str] = Field(None, description="Queue tag for reference geometry tasks")
    dft_optimization_program: str = Field("psi4", description="Program for DFT geometry optimizations")
    qc_keywords: Optional[Dict[str, Any]] = Field(
        None,
        description=QC_KEYWORDS_DESCRIPTION + " Applied to every DFT geometry optimization spec.",
    )
    dft_optimization_keyword: SkipJsonSchema[Optional[Any]] = Field(
        None, description="Deprecated, use qc_keywords.", exclude=True,
    )
    _dep_dft_optimization_keyword = deprecated_null_only("dft_optimization_keyword", "qc_keywords")
    tag_dft_geometry: Optional[str] = Field(None, description="Queue tag for DFT geometry tasks")
    tag_nl_geometry: Optional[str] = Field(
        None,
        description=(
            "Queue tag for the VV10 '-V' (ORCA) tasks when "
            "include_nl_functionals is set. ORCA typically runs on a "
            "separate manager, so point this at that manager's tag. "
            "Falls back to tag_dft_geometry when unset."
        ),
    )
    use_initial_reference_geometry: bool = Field(False, description="Use initial (unoptimized) reference geometry")
    trajectory_analysis: bool = Field(
        True,
        description=(
            "If True (default), evaluate each DFT functional via SP+gradient "
            "at every geometry along the reference optimization trajectory "
            "and report RMSD of the per-component force (meV/Å) vs the "
            "reference. Combined with the equilibrium-geometry RMSD via a "
            "z-score-weighted score (see ``score_weights``). Absolute "
            "energies are not used here — use the energy_benchmark workflow "
            "for relative-energy comparison. Set to False to keep the "
            "legacy eq-geometry-only behaviour."
        ),
    )
    trajectory_dedup_rmsd: Optional[float] = Field(
        None,
        description=(
            "Compute-saving pre-filter for the trajectory force benchmark. "
            "If set (Angstrom, e.g. 0.02), reference-trajectory frames are "
            "grouped by Kabsch-aligned RMSD and only geometrically distinct "
            "frames (RMSD > this value from the last kept frame) are kept, so "
            "SP+gradients are not computed at near-identical copies (the "
            "optimizer's stationary tail). Applied to the shared reference "
            "frame set, so reference and every functional stay aligned. The "
            "ranking is essentially unchanged; it only reduces cost. None or "
            "0 (default) disables dedup."
        ),
    )
    trajectory_dedup_min_frames: int = Field(
        1,
        description=(
            "Per-structure floor on frames kept by trajectory_dedup_rmsd, so "
            "a near-stationary system still contributes at least this many "
            "points (the first frame plus the most-displaced ones). Ignored "
            "when trajectory_dedup_rmsd is unset."
        ),
    )
    fragments: Optional[Dict[str, List[List[int]]]] = Field(
        None,
        description=(
            "Optional, used by the trajectory force benchmark. "
            + FRAGMENTS_DESCRIPTION
            + " When given, the force error at every trajectory step is "
            "split per fragment into its rigid-body part (net force and "
            "torque on the fragment, which by Newton's third law is purely "
            "intermolecular) and the intramolecular remainder, so the stiff "
            "bond-stretch errors that dominate the total force RMSD can be "
            "separated from the intermolecular ones. Without it only the "
            "total per-component force RMSD is reported."
        ),
    )
    ranking_metric: Literal["total", "intermolecular"] = Field(
        "intermolecular",
        description=(
            "Force metric that enters the combined z-score ranking as "
            "``rmsd_force`` (together with ``rmsd_eq``) and sorts the "
            "decomposition table. 'intermolecular' (default) = RMSD of the "
            "net force per fragment, the part relevant to binding; requires "
            "``fragments`` and falls back to 'total' with a warning when "
            "they are absent. 'total' = RMSD over every Cartesian force "
            "component (the pre-2026-11 behaviour), dominated by stiff bond "
            "stretches."
        ),
    )
    score_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "rmsd_eq": 1.0,
            "rmsd_force": 1.0,
        },
        description=(
            "Weights for the combined z-score ranking when "
            "trajectory_analysis is enabled. Keys: rmsd_eq, rmsd_force. "
            "Default: equal weighting."
        ),
    )
    generate_plots: bool = Field(
        False,
        description=(
            "If True, also produce eq-RMSD histograms and (when "
            "trajectory_analysis is enabled) trajectory force-error "
            "histograms + violin plot as SVG under data/plots/. Off by "
            "default — JSON results are always written."
        ),
    )

    include_nl_functionals: bool = Field(
        False,
        description=(
            "Also benchmark VV10 '-V' functionals (B97M-V, WB97M-V, WB97X-V "
            "at def2-tzvpd). These are routed to the ORCA driver with SCNL "
            "(program=orca, keywords={'simple_input': 'SCNL'}), regardless of "
            "dft_optimization_program, because Psi4 has no analytic VV10 "
            "gradient. Off by default."
        ),
    )

    @model_validator(mode="after")
    def _validate_fragments(self):
        if self.fragments is not None:
            validate_fragment_partition(self.fragments, self.benchmark_structures)
        return self

    @field_validator("reference_geometry_level_of_theory")
    @classmethod
    def _lower_ref_geom_lot(cls, v):
        """Lowercase method (index 0) and basis (index 1); leave program (index 2) as-is."""
        if not isinstance(v, list):
            return v
        out = list(v)
        for i in (0, 1):
            if i < len(out) and isinstance(out[i], str) and out[i]:
                out[i] = out[i].lower()
        return out
