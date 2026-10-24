"""Sampling workflow for adsorbate binding sites on periodic slabs (MLP-only)."""
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, model_validator
from .base import ServerConfig, LevelOfTheory


class SamplingPeriodicConfig(BaseModel):
    """Grid-based sampling of adsorbate binding sites on a periodic slab.

    Runs one MLP-driven optimization per candidate placement. There is no
    separate refinement stage — the same LOT drives the initial and only
    optimization pass. The slab may be partially frozen (typical use:
    freeze the bottom layers, let the top surface + adsorbate relax).
    """
    workflow: Literal["sampling_periodic"] = Field(..., description="Must be 'sampling_periodic'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")

    # Adsorbate + slab data sources
    molecule: str = Field(..., description="Adsorbate name in the small_molecule_collection")
    small_molecule_collection: str = Field("Small_molecules", description="Adsorbate lookup dataset")
    atoms_collection: str = Field("atoms", description="SinglepointDataset with atomic adsorbates")
    surface_collection: str = Field(..., description="OptimizationDataset holding periodic slab structures")
    surface_clusters: List[str] = Field([], description="Subset of slabs to sample (empty = all entries in surface_collection)")
    surface_specification: Optional[str] = Field(
        None,
        description=(
            "Optimization specification in surface_collection whose result is the slab: each "
            "slab is the final molecule of that spec's (complete) record, e.g. the slab relaxed "
            "with the sampling model, the cell still from the entry's extras. None (default) "
            "takes the entry's initial molecule."
        ),
    )

    # Periodic cell (per-slab override)
    cell: Optional[List[List[float]]] = Field(
        None,
        description=(
            "3x3 cell vectors in Angstrom applied to every slab. If None, the "
            "cell is read from surface.extras['cell'] on each slab record."
        ),
    )
    pbc: List[bool] = Field(
        [True, True, False],
        description="Periodic axes (default = 2D slab: pbc in x, y; vacuum in z).",
    )

    # Grid
    step_size_ang: float = Field(3.0, description="Grid spacing in Angstrom")
    grid_noise_frac: float = Field(
        0.25,
        description="Random ± jitter per interior grid node as fraction of step_size_ang. Set 0.0 to disable.",
    )
    grid_offset: bool = Field(
        False,
        description=(
            "Shift the sampling lattice by a random amount in [0, step) along x and y, drawn "
            "from random_seed. With a new seed this samples new positions; without it, seeds "
            "only differ by the jitter of at most grid_noise_frac * step."
        ),
    )

    # Placement
    sampling_distance_ang: float = Field(
        2.5, description="Target adsorbate-to-nearest-surface-atom distance in Angstrom"
    )
    cavity_z_scan_step_ang: float = Field(
        0.5, description="Z-scan step for cavity placement in Angstrom"
    )
    cavity_z_scan_window_ang: float = Field(
        1.0,
        description=(
            "Accept z where the nearest-atom distance is in "
            "[sampling_distance - window, sampling_distance]. Pick best-fit z. "
            "Larger values catch deeper cavities but risk placing above the surface."
        ),
    )

    # Sanity checks
    sanity_min_distance_ang: float = Field(
        1.5,
        description="Every adsorbate atom must be at least this far from every surface atom (Angstrom).",
    )
    sanity_max_iter: int = Field(
        20,
        description="Max random-rotation attempts per grid point before skipping the site.",
    )

    # Reproducibility
    random_seed: Optional[int] = Field(
        None, description="RNG seed for grid noise + rotation. None = non-reproducible."
    )

    # Level of theory (MACE only for now — enforced by validator)
    sampling_level_of_theory: LevelOfTheory = Field(
        ..., description="MACE level of theory for the periodic optimizations"
    )

    # Optimization keywords (merged over the default {'maxiter': 125})
    sampling_opt_keywords: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Extra geomeTRIC keywords; merged over the built-in {'maxiter': 125}. "
            "Leave 'coordsys' at geomeTRIC's default 'tric': 'cart' is rejected by "
            "geomeTRIC whenever slab freezing is on ('Do not use constraints with "
            "Cartesian coordinates'), and on large slabs it converges to visibly "
            "wrong geometries."
        ),
    )

    # Duplicate-site filtering
    site_filter: str = Field(
        "periodic",
        description=(
            "Duplicate-site filter. 'periodic' compares adsorbate position under the "
            "minimum-image convention plus an element-resolved height profile: correct "
            "for a cell (the cluster filter has no notion of one, so it keeps every "
            "wrap-around duplicate) and ~0.15 s for 120 structures against hours, since "
            "it does not Kabsch-align 1500-atom slabs whose frozen layers are identical "
            "by construction. 'cluster' uses the original filter_binding_sites."
        ),
    )
    orientation_tol_ang: Optional[float] = Field(
        0.3,
        description=(
            "Height-profile tolerance (Angstrom) for the periodic filter. Two sites at "
            "the same position count as one only if their per-element sorted heights "
            "above the adsorbate COM agree within this. Keeps physically distinct modes "
            "apart (C-down vs O-down CO ~1.1 A; face- vs vertex-down CH4 ~0.8 A) while "
            "ignoring non-modes (atom relabelling, methyl torsion: exactly 0). "
            "Set null to compare positions only."
        ),
    )

    # Optimization engine
    sampling_opt_program: str = Field(
        "geometric",
        description=(
            "QCEngine optimization procedure driving the geometry relaxations. "
            "'geometric' uses internal coordinates: better convergence per step, but its "
            "Wilson-B / G-matrix work is O(N^2-N^3) and dominates for large slabs (~200 s "
            "per step at 1500 atoms, against a ~1 s MLP gradient). 'ase' uses preconditioned "
            "Cartesian optimizers at ~0.1 s per step and handles frozen atoms as a force "
            "mask. Prefer 'ase' for periodic slabs and other large systems; 'geometric' "
            "remains better for expensive gradients on normal-sized systems."
        ),
    )

    # Slab freezing
    freeze_below_z_ang: Optional[float] = Field(
        None,
        description=(
            "Freeze every slab atom whose z-coordinate (Angstrom, in the slab's own frame) "
            "is below this value during optimization. None = fully relaxed."
        ),
    )
    freeze_atoms: Optional[List[int]] = Field(
        None,
        description=(
            "Explicit 0-indexed atom list to freeze in the combined slab+adsorbate "
            "molecule. Overrides freeze_below_z_ang when both are set."
        ),
    )

    # Filter
    rmsd_value: float = Field(
        0.40,
        description=(
            "Duplicate-site tolerance (Angstrom). With site_filter='periodic', the adsorbate "
            "centre-of-mass distance (minimum image); with 'rmsd', the RMSD threshold."
        ),
    )
    rmsd_symmetry: bool = Field(False, description="Account for molecular symmetry in RMSD comparison")

    # Compute + output
    sampling_tag: str = Field("sampling_periodic", description="Queue tag for the optimization tasks")
    store_initial_structures: bool = Field(
        False, description="Write pre-optimization xyz files under data/site_finder/ for debugging"
    )
    dataset_suffix: str = Field(
        "",
        description=(
            "Suffix of this sampling run's datasets: complexes in '<mol>_<slab><suffix>', "
            "bare-surface references in '<mol>_<slab><suffix>_surface'. Entries are keyed by "
            "site name and keep the geometry of the run that created them, so every run with "
            "different geometries (another seed or grid, another sampling model) needs its own "
            "suffix (e.g. '_v1'). Reusing an entry with a different geometry raises. "
            "Default '' keeps the historical names."
        ),
    )
    resume_from_existing: bool = Field(
        False,
        description=(
            "Treat the entries already stored in '<mol>_<slab><suffix>' as the run: optimize "
            "any that are missing, then continue (unique-site filter, bare-surface references) "
            "without adding newly generated candidates. Use it to finish or extend a run, e.g. "
            "add bare-surface references later, when the candidate generator of the current "
            "BEEP version no longer reproduces the stored placements. Slabs without stored "
            "entries are sampled normally."
        ),
    )
    bare_surface_references: bool = Field(
        True,
        description=(
            "Optimize one bare-surface reference per unique site (the adsorbate "
            "stripped from the relaxed complex) into '<mol>_<slab><suffix>_surface'. "
            "They are needed only for binding energies: an active-learning round "
            "that consumes the sampling trajectories can set this to false and "
            "skip roughly half of the optimizations. be_comp_periodic processes "
            "only sites present in both datasets, so the references can be added "
            "later by re-running sampling with the flag back on and "
            "resume_from_existing = true."
        ),
    )

    @model_validator(mode="after")
    def _require_mace_and_pbc_shape(self):
        if not self.sampling_level_of_theory.is_mace:
            raise ValueError(
                "sampling_periodic requires an MLP level of theory "
                "(set 'mace_model' in sampling_level_of_theory)."
            )
        if len(self.pbc) != 3:
            raise ValueError(f"pbc must be length 3 (got {len(self.pbc)}).")
        if self.cell is not None:
            if len(self.cell) != 3 or any(len(row) != 3 for row in self.cell):
                raise ValueError("cell must be a 3x3 list of lattice vectors in Angstrom.")
        if not 0.0 <= self.grid_noise_frac <= 0.5:
            raise ValueError(
                f"grid_noise_frac must be in [0, 0.5] (got {self.grid_noise_frac})."
            )
        coordsys = (self.sampling_opt_keywords or {}).get("coordsys")
        freezing = self.freeze_below_z_ang is not None or self.freeze_atoms is not None
        if (
            self.sampling_opt_program == "geometric"
            and freezing
            and coordsys is not None
            and str(coordsys).lower() == "cart"
        ):
            raise ValueError(
                "sampling_opt_keywords {'coordsys': 'cart'} cannot be combined with slab "
                "freezing (freeze_below_z_ang / freeze_atoms): geomeTRIC raises "
                "'Do not use constraints with Cartesian coordinates'. Drop 'coordsys' to "
                "use the default 'tric', which is also the numerically reliable choice on "
                "large slabs."
            )
        return self
