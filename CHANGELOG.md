# Changelog

All notable changes to BEEP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — 0.16.0.dev

### Fixed

- **`pre_exp`: atoms got a zero prefactor.** With all three moments of
  inertia zero, `is_linear_rotor(0, 0)` is false and the nonlinear branch
  multiplied by `sqrt(Ia Ib Ic) = 0`, so `pre_exponential_factor` returned
  `nu = 0` for H, C, N, O. Atoms are now detected (`is_atom`, all moments
  below 1e-50 kg m^2) and take `q_rot = 1`, so `nu` reduces to the
  translational term. Test added.

- **`be_comp_periodic`: the gas-phase adsorbate must be optimized at
  the geometry level, as in `be_hess`.** `gas_adsorbate()` looked for
  the adsorbate optimized at the *electronic* LOT and, when the
  small-molecule collection had no such spec (the usual case for an
  MLP), silently fell back to the entry's input geometry. The BE/DE
  reference was then not a minimum of the potential and every BE
  shifted by the same amount (CO with the MPWB1K input geometry under
  a B3LYP-D4 MACE model: r(CO) 1.114 vs 1.125 Å, BE too strong by
  0.15 kcal/mol at every site; IE unaffected). For `quantity` 'be' and
  'all' the adsorbate is now read at `opt_level_of_theory`, the level
  of the complexes and bare surfaces, and a missing spec or an
  incomplete record raises with "please optimize it first". 'ie' only
  needs the adsorbate's charge and multiplicity and still uses the
  entry geometry; single atoms need no optimization.
- **`sampling`: relaunches resume instead of re-sampling finished
  clusters.** `run()` walked the clusters in order and ran a full
  placement + gfn2 round on each one before checking the target, so a
  relaunch after a crash or a timeout re-sampled every finished cluster
  (a whole gfn2 batch per cluster, often hitting the 24 h limit again
  before reaching the remaining ones), and a running workflow never
  revisited a cluster whose sampling optimizations finished or were
  reset after it had moved on. `run()` is now three phases: a pre-scan
  that opens every cluster's `pre_<mol>_<w>` / `<mol>_<w>` datasets and
  logs a resume table (sampled / complete / refined per cluster); a
  harvest pass (new `harvest_cluster`) that waits for pending sampling
  optimizations, filters the completed ones not yet in the refinement
  dataset and resubmits refinement for the new unique sites, without
  generating any placements; and a top-up pass that runs the existing
  per-cluster placement loop only while the global refined count is
  still below `total_binding_sites`, visiting never-sampled clusters
  first and topping up already-sampled ones only if the target is still
  unmet after every cluster has had one round (a second round on a
  sampled cluster mostly rediscovers minima the filter already
  removed). The sampling summary now covers
  all clusters and reports how many sites each one contributed by
  harvesting. Datasets and entry names are unchanged, so existing runs
  resume as they are. Regression tests cover the no-resample, harvest-
  counts-toward-target, top-up-stops-at-target and harvest-skips-
  refined cases.

- **`energy_benchmark`: results are now reported per optimization
  geometry.** Reaction entries have always been per `(structure,
  opt_level_of_theory)` pair, but the printed MAE summary pooled every
  geometry's errors into one unlabeled table — with several opt LOTs
  each functional's MAE was an average over all geometry sets, and even
  with one LOT the output never said which geometry it belonged to. The
  BE and gCP-corrected summaries now print one full per-category section
  per entry in `opt_level_of_theory`, each labeled with its geometry LOT
  and computed only from that LOT's binding sites (new
  `log_mae_per_geometry` helper, suffix-matched on the entry names). A
  LOT with no completed entries warns and is skipped. Regression tests
  cover per-section MAE separation and the skip path.

- **`energy_benchmark`: open-shell CCSD(T)/CBS records no longer die on
  the DF-CCSD iteration cap.** On open-shell systems (seen on CN and
  CH3O) DIIS can oscillate around the residual criterion with the energy
  already converged to ~1e-7 Eh, and the default `cc_maxiter` then kills
  a converged calculation. `get_cc_keywords` now sets `cc_maxiter: 200`
  in the open-shell branch; the closed-shell branch is unchanged.
  Existing CBS specs on the server keep their stored keywords — repairing
  the already-failed records needs the spec-keyword update plus
  cancel→uncancel→reset on those records (documented procedure).

- **`mbe_extract`: ZPVE-corrected MBE binding energies were too large by
  2·ΔZPVE.** MBE BEs are positive-convention (complex minus fragments,
  negated) and `Delta_ZPVE` = ZPVE(complex) − ZPVE(parts) is positive, so
  the correction has to be subtracted. The `+` had been carried over from
  `extract`, where BEs are negative-convention and `+` is right. The
  regression test that locked in the wrong sign now checks it with
  asymmetric numbers.

- **`extract` / `apply_lin_models`: `StdDev_all_dft` was taken over the
  method columns plus the freshly inserted mean.** The mean adds zero
  deviation but inflates the denominator, so the reported spread was low by
  √((n−1)/n), about 10 % for five functionals. Both are now computed over
  the method columns only, as the `--be_methods` path already did.

- **`sampling_periodic`: `random_seed` did not control the adsorbate
  orientation.** qcelemental's `Molecule.scramble` draws its rotation from
  numpy's global RNG, not Python's `random`, so only the grid noise was
  reproducible. The rotation matrix is now built from the workflow's own
  seeded RNG and passed to `scramble` explicitly.

- **`sampling_periodic`: adsorbates placed on the x = 0 / y = 0 grid lines
  were split across the cell faces.** The per-atom wrap ran before COM
  recentering, and the naive mean of the split coordinates put the "centre"
  mid-cell, so the stored entry kept the split molecule and the duplicate
  filter used the same wrong COM. Atoms are now unwrapped (minimum image
  from the first adsorbate atom) before the COM is taken and wrapped once
  after centring; the duplicate filter uses the same unwrapped COM.

- **`sampling` (sphere method): any triatomic adsorbate had diameter 0.**
  `calculate_diameter` special-cased a `(3, 3)` array as "one water
  molecule", but it is called on the adsorbate, so HCO, H2O, HCN and CO2
  silently lost the inter-candidate spacing check. `single_site_spherical_sampling`
  also converted the shell radius to bohr twice (1.89× the requested radius).

- **Open-shell adsorbates lost their multiplicity in MBE fragmentation and in
  the sampled complexes.** `mbe_fragmentation` hard-coded charge 0 /
  multiplicity 1 for every fragment, so HCO, CH3O and CH2OH were submitted as
  singlets; `sampling` and `sampling_periodic` rebuilt the complex without
  fragment charges/multiplicities and let qcelemental infer a parity-based
  value. Complexes now carry explicit fragments with their own charge and
  multiplicity (total charge summed, unpaired electrons coupled high-spin).

- **`be_hess`: cluster names were re-parsed from entry names and broke for
  anything but `Wnn_mm`.** `split("_")[-3:-1]` gave `CO_cd5` for a cluster
  named `cd5`, so the reaction dataset was looked up under the wrong name and
  `compute_hessian` silently submitted nothing. The cluster name is now
  carried from `check_refinement_status`; a cluster emptied by the RMSD
  filter is skipped with a warning instead of raising `IndexError`.

- **`compute_hessian` / `compute_be_dft_energies` crashed on basis-less
  levels of theory** (`gfn2-xtb`, MACE aliases) with a two-way
  `split("_")`; only the first underscore separates method from basis now.

- **`compute_hessian` reuse pre-query ignored program and dispersion.** A
  stray psi4 Hessian at the same functional/basis marked an ORCA or Gaussian
  molecule as done (and vice versa), suppressing submission while
  `get_zpve_mol` later reported "no Hessian". Reuse is now scoped to the
  requested program and dispersion keyword.

- **MBE monitors polled forever on cancelled records.** Only COMPLETE and
  ERROR were terminal and `max_wait` defaults to `None`; CANCELLED, INVALID
  and DELETED are now terminal failures, logged and counted.

- **`nm_sampling` could loop forever when `client.reset_records` kept
  failing**; failed reset calls now consume the per-record retry budget.
  An aborted run (no geometries, modes or displacements) now exits 1
  instead of logging "finished successfully".

- **Silently discarded QC keywords.** The `nm_sampling`, `geom_benchmark`
  and `sampling` program-keyword fields (now `qc_keywords`, see Changed)
  were typed as QCFractal 0.15 keyword IDs and every consumer replaced a
  non-dict with `{}`, so user SCF options vanished without a trace.
  `qc_keywords` is an inline dict that reaches the specification; the
  adapter raises `TypeError` on a non-dict instead of discarding it, and a
  legacy ID under any of the old names is rejected at config validation.

- **Specification/entry insertions no longer hide server errors.**
  `add_specification` / `add_entry` return `InsertMetadata`; a same-name
  spec with different content was silently ignored and the old spec ran.
  Every insertion site now logs the server's error list.

- **`be_hess` / `energy_benchmark` monitoring waited on every level of theory
  ever registered on the reaction datasets**, not only the specs submitted
  in the current run, so ERRORs from old LOTs stalled and polluted the
  status. `_collect_reaction_record_ids` is now filtered by spec name.

- **`create_or_load_reaction_dataset` crashed on WAITING/RUNNING
  optimizations** (`final_molecule` is `None`); only COMPLETE records become
  reaction entries and the entry count reflects successful additions.

- **`check_optimized_molecule` raised a raw `PortalRequestError`** for a
  missing entry instead of `LevelOfTheoryNotFound`.

- **`rmsd_filter_from_dataset` dropped ERROR records silently and kept exact
  duplicates** (`rmsd != 0.0`); both now behave as the docstring says.

- **`geom_benchmark` crashed on `record.id` when a LOT had no record for a
  structure** (qcportal 0.64 returns `None`); the LOT is excluded with a
  warning.

- **`extract` ZPVE guard counted BE rows, not finished Hessians**, so the
  linear model could be fitted from fewer than five points; the process-wide
  `warnings.filterwarnings("ignore")` that hid the resulting `RankWarning`
  (also in `energy_benchmark`) is gone. `extract` no longer raises `KeyError`
  when a `be_nocp` entry has no BSSE row (3c methods), and `zpve` no longer
  raises `TypeError` when the Hessian record has no electronic energy.

- **`pre_exp`: prefactors were 10x too large by default and 1.77x too large
  for linear molecules.** The `molecule_surface_area` default was written
  `10e-19`, which is 1e-18 m²; the help text and Minissale et al. 2022 (A
  fixed to 10⁻¹⁹ m² for all molecules) intend 1e-19 m², and ν is linear in
  A. The linear-rotor branch implemented Minissale et al. 2022 Eq. 20,
  `sqrt(pi)/(sigma h^2) * 8 pi^2 kB T * I`, which carries a `sqrt(pi)` left
  over from the three-dimensional orientation volume (their Table 4 values
  carry a further factor `pi`); the classical linear rotor is
  `8 pi^2 I kB T / (sigma h^2)`, which reproduces the spectroscopic
  high-temperature limit `kB T / (sigma h c B)` (CO at 298 K: 107.3 for
  B = 1.9313 cm⁻¹, versus 190 for Eq. 20 and 337 for Table 4). The
  nonlinear branch and the translational part were already correct and
  reproduce Table 4 to four digits. Tests pin the linear branch to the CO
  spectroscopic value and the nonlinear one to Minissale Table 4.

- **`pre_exp`: `molecule: []` crashed, the temperature range excluded
  `T_max` although the log said inclusive, and linear molecules were
  detected with an exact `Ia == 0` float compare.** `[]` now means all
  molecules, the range is inclusive, `range_of_temperature` is validated,
  and linearity uses a relative tolerance (`Ia < 1e-6·Ib`).

- **`sampling` with `store_initial_structures` mangled any path containing a
  dot** (`.replace(".", "")` ran on the whole path); the early stop fired
  one cluster late (`>` instead of `>=`); the fallback to the `n_water//3`
  slot heuristic was a silent bare `except` and now warns with the cause.

- **`sampling_periodic` read the adsorbate through an optimization record at
  the MACE spec**, which MLP-only runs never have; it now uses the entry's
  initial-molecule slot like the slab side.

- **`cbs_extrapolation`: `scf_xtpl_helgaker_2` raised `NameError` for array
  input with `verbose > 2`**; Halkier 1998 DOI corrected.

- **`beep --schema` used the removed pydantic v1 `schema_json`; an invalid
  config now prints a per-field error list and exits 1** instead of a
  traceback. Importing `beep` from an uninstalled checkout no longer raises
  `PackageNotFoundError` (`__version__` falls back to `0.0.0+unknown`).

- **`mbe`: the `add_entry(overwrite=...)` compatibility shim swallowed every
  `TypeError`**, including molecule validation errors; it now re-raises
  anything but the signature mismatch.

- **`extract`: methods present only in later clusters no longer vanish
  from the report.** `concatenate_frames` reindexed every subsequent
  cluster's frame onto the columns accumulated from the first-iterated
  cluster, silently projecting away any method that cluster lacked — so
  a functional (or MACE model) added mid-campaign disappeared from the
  extraction whenever a method-poor cluster happened to iterate first,
  making results dependent on cluster iteration order. The reindex is
  gone; `pd.concat`'s outer join takes the column union, and clusters
  lacking a method carry NaN rows (downstream means are skipna).
  Regression test covers both iteration orders.

- **`sampling_periodic`: every frozen-slab optimization died on submission.**
  `build_freeze_constraint_string` emitted geomeTRIC's *classic text* constraints
  block (`"$freeze\nxyz 1-3\n"`), but the JSON API that QCEngine drives
  (`geometric.run_json`) expects the *structured* form and renders that text
  itself — so it called `.items()` on a `str` and raised
  `AttributeError: 'str' object has no attribute 'items'` before the first
  gradient. Every `sampling_periodic` run with `freeze_below_z_ang` /
  `freeze_atoms` set (i.e. every realistic slab run) failed 100% at step zero.
  Replaced by `build_freeze_constraints`, returning
  `{"freeze": [{"type": "xyz", "indices": [...]}]}`. Indices stay **0-based**:
  geomeTRIC's `commadash` does the 0->1 shift and range compression itself
  (`[0,1,2] -> "1-3"`), so the frozen atom set is unchanged — only the handoff
  format is fixed. User-supplied `constraints` are now merged as dicts, and a
  string raises a clear `ValueError` instead of the geomeTRIC `AttributeError`.
  New regression test feeds our output through geomeTRIC's own
  `make_constraints_string` and requires the classic block back.
  (Removed the now-unused `_atom_index_ranges` helper.)

- **`sampling_periodic`: `coordsys: "cart"` is incompatible with slab freezing.**
  geomeTRIC raises `RuntimeError: Do not use constraints with Cartesian
  coordinates`, so the combination could never run — yet
  `examples/sampling_periodic.json` shipped `{"coordsys": "cart"}` alongside
  `freeze_below_z_ang`, and the config docstring recommended cart for large
  slabs. The example now leaves `coordsys` at geomeTRIC's default `tric`, the
  field description no longer recommends cart (it is also numerically
  unreliable on large slabs), and a config validator rejects the combination at
  load time with an actionable message instead of failing mid-run.

- **`be_comp_periodic`: periodic D3 was routed to a non-periodic harness.**
  `DISPERSION_PROGRAMS` maps `-d3bj` to QCEngine's legacy `dftd3` harness, which
  wraps the standalone executable and has no periodic support: handed
  `keywords['cell']`/`['pbc']` it ignores them and returns *cluster* dispersion
  for a slab, with no error. Added `PERIODIC_DISPERSION_PROGRAMS` /
  `periodic_dispersion_program()` mapping `dftd3 -> s-dftd3` (the python-API
  harness that honours cell/pbc; `-d4` already routed to the python-API
  `dftd4`), applied in `be_comp_periodic` to the periodic *and* gas-phase specs
  so the `BE = complex - surface - gas` difference cancels within one harness.
  Scoped to the periodic path on purpose: cluster workflows keep `dftd3`, so
  their existing dispersion specs and records stay valid.

### Changed

- **`be_comp_periodic`: separate queue tag for the dispersion.** New optional
  `disp_tag`: the dispersion reactions (analytic periodic D4/D3, CPU) are
  submitted under it, the electronic (MACE, GPU) ones under `be_tag`. Before,
  both went to `be_tag`, so a GPU tag also carried CPU work and the two kinds of
  manager could only be kept apart by the programs their environments
  advertise. Default `None` keeps one tag.

- **`be_comp_periodic` / `be_assemble_periodic`: one ReactionDataset per
  surface family.** New optional `surface_family` (e.g. `"npASW"`): the
  reactions of all slabs go to `<smol>_<family><suffix>_be` / `_ie` / `_de`
  instead of one dataset per slab. Per-slab datasets were needed only because
  the non-periodic axis is padded per slab, so every slab had its own cell
  keywords and specification; the slabs of a family share the lateral cell and
  the non-periodic axis is now padded to the largest slab's (any padding above
  a slab's own is equally valid). Slabs with another lateral cell raise. Entry
  names already carry the slab; the assembly splits them back per slab and
  writes the same CSVs. Default `None` keeps one dataset per slab.

- **`sampling_periodic`: a deep cavity-scan best fit that is covered from
  above goes to the top of the run.** A column running down the wall of a
  narrow channel can stay inside the scan window for several Angstrom; its
  best fit could then be the deep end, which placed CO in enclosed cavities
  (~5 A under the surface, water above, positive frozen-fragment IE). If the
  slab atoms above the best fit (within 3.5 A laterally) surround it with no
  azimuthal gap of 120 deg or more, the candidate goes to the top of the run;
  open channels keep their pore sites.

- **`be_comp_periodic` / `be_assemble_periodic`: BE, IE and DE as
  ReactionDatasets.** The periodic energies were single points in
  separate datasets (`<smol>_<slab>_be_sp`, `_surface_be_sp`,
  `<smol>_gas_be_sp`, and for the IE `_ie_slab_sp` / `_ie_ads_sp`),
  combined only in the assembly code. Each quantity is now a
  ReactionDataset per slab whose entries carry the site's stoichiometry
  (new `periodic_stoichiometry`): `<smol>_<slab>_be` (complex - relaxed
  bare surface - gas adsorbate), `_ie` (complex - slab - adsorbate,
  frozen at the complex geometry) and `_de` (frozen fragments - relaxed
  references), so BE = IE + DE holds on the server. `quantity` 'be' and
  'ie' build one dataset, 'all' all three. Every component uses the same
  periodic specification, the gas-phase adsorbate included; a component
  shared between datasets is one record. The assembly reads the reaction
  records and their components and writes the same CSVs as before.
  Datasets from earlier runs are not read; rerunning `be_comp_periodic`
  rebuilds them from the existing optimizations and reuses the existing
  periodic single-point records.

- **Program keywords are `qc_keywords` in every workflow config.** The
  per-program QC options (psi4/ORCA/Gaussian keywords such as `guess`,
  `damping_percentage`, `scf_type`, `reference`) that become a
  specification's `qc_spec.keywords` were exposed under one name per
  workflow, all of them misleading: `keyword_id` (`sampling`, `be_hess`,
  `energy_benchmark`), `dft_keyword` (`nm_sampling`) and
  `dft_optimization_keyword` (`geom_benchmark`). They are now a single
  `qc_keywords: dict | null` field, distinct from the `*_opt_keywords`
  fields, which are geomeTRIC optimizer options. `sampling.qc_keywords`
  goes verbatim to the refinement `qc_spec`; `be_hess.qc_keywords` is
  merged over the open-shell default `{"reference": "uks"}` for the BE
  single points; `nm_sampling.qc_keywords` reaches every DFT gradient spec
  and `geom_benchmark.qc_keywords` every DFT optimization spec. Motivation:
  the UKS SCF of the F atom (²P) on water oscillates with the default SAD
  guess at every site (`{"guess": "gwh", "damping_percentage": 20}` fixes
  it), and there was no consistent way to pass SCF keywords through the
  configs. The old names remain as deprecated aliases that accept `null`
  only, so existing configs carrying `"keyword_id": null` etc. still load;
  any other value (a legacy QCFractal 0.15 keyword ID or a dict under the
  old name) fails validation with a message pointing to `qc_keywords`.
  `energy_benchmark.keyword_id` was never read by the workflow and has no
  replacement. The deprecated fields are excluded from the config copy
  written to disk and from `beep --schema`. A config that sets a dict
  creates a new specification and must not be re-run on species already
  computed with the default one.

- **`examples/*.json` are now complete, validating configs** (previously six
  of them carried `null` in required fields and failed model validation);
  a test loads every example against its workflow model.

- **Removed `BeHessConfig.dispersion_tag`** (added in 0.15.0 alongside
  `mace_dispersion`). It routed the analytic dispersion single-points to
  a separate queue tag from the MLP energies, but QCFractal already
  matches tasks to managers on *program availability* at claim time: a
  GPU manager without `dftd4`/`s-dftd3` in its environment never claims
  dispersion jobs, and a CPU manager without `mace` never claims MLP
  jobs — so a single `energy_tag` routes both spec families correctly
  and the extra knob solved a problem that doesn't exist with disjoint
  worker environments. Configs that set `dispersion_tag` must drop the
  field; dispersion submissions now always use `energy_tag`.

- **`geom_benchmark` is now entry-based (BREAKING config change),
  mirroring the `nm_sampling` generalisation.** The workflow benchmarks
  exactly the entries listed in `benchmark_structures`, all living in a
  single new required `opt_dataset` OptimizationDataset. All
  adsorbate/surface special-casing is gone:
  - **Removed fields**: `molecule`, `small_molecule_collection`,
    `surface_model_collection`, `atoms_collection`, `bsse_test` (and the
    `BSSETestConfig` model).
  - **Removed behavior**: the implicit adsorbate-monomer reference
    optimization, the bare-surface benchmarking derived from entry-name
    parsing, the atom-adsorbate fallback, the global adsorbate
    multiplicity gate (each stored molecule carries its own
    charge/multiplicity), and the entire direct-Slurm counterpoise BSSE
    test (~250 lines of non-QCFractal-native code:
    `_write_cp_python_script`, `_write_slurm_script`, `_submit_cp_jobs`,
    `_wait_for_cp_jobs`, `_collect_cp_results`).
  - **To benchmark a monomer or bare surface alongside the complexes**,
    add it as another entry in `opt_dataset` — every entry is on equal
    footing.
  - Entry names are free-form; the `rsplit("_", 1)`
    dataset-name-in-entry-name convention no longer applies.
  - Output folder + log/config names derive from `opt_dataset` instead
    of `molecule`; the trajectory-analysis plots are named after the
    dataset as well.
  - `trajectory_analysis` remains optional (default `True`,
    `false` = legacy eq-geometry-RMSD-only run).
  - Net: `workflows/geom_benchmark.py` shrinks from 846 to 386 lines.

- **`nm_sampling` generalised from 2 fragments to N (BREAKING config change).**
  The mode classifier previously hardcoded "last `n_adsorbate_atoms` = adsorbate,
  rest = cluster" — silently mislabelling intermolecular modes of any non-last
  fragment as bending on N-mer systems (N>2). It now takes an explicit
  `fragments: List[List[int]]` partition (matches `qcel.Molecule.fragments`
  shape) and sums the per-fragment rigid-body-motion projection over every
  fragment. The N=2 case (`fragments=[[cluster], [adsorbate]]`) reproduces
  the old behavior exactly.

  Config changes (breaking, no deprecation alias):
  - **New required** `opt_dataset: str` — the OptimizationDataset containing
    every entry in `benchmark_structures`. Entry names are now free-form
    labels; no rsplit / dataset-name-in-entry-name convention.
  - **New required** `fragments: Dict[str, List[List[int]]]` — per-
    benchmark-structure partition, keys must exactly cover
    `benchmark_structures`. Each value is a list of 0-indexed atom-index
    lists (matches `qcel.Molecule.fragments` shape); every atom must
    appear in exactly one fragment. Per-structure partitions are
    required because collapsing multiple monomers into a single rigid
    block mislabels intra-block rearrangements (the same failure mode
    the N-fragment classifier fixes at the physics layer). Use one
    fragment per monomer.
  - **Removed**: `molecule`, `small_molecule_collection`,
    `surface_model_collection`. None of them influenced the computation
    once the fragment partition is explicit; the old adsorbate-atom-count
    lookup that motivated them is gone, and output naming derives from
    `opt_dataset` instead of `molecule`.

  Migration for existing configs (BEEP-1 cluster+adsorbate case): add
  `opt_dataset` (was implicit via `benchmark_structures` name mangling),
  add `fragments: [[0..n_cluster-1], [n_cluster..n_total-1]]`, remove the
  three old fields. `nm_sampling` output folder is now `<opt_dataset>/`
  instead of `<molecule>/`. Existing OptimizationDatasets on the server
  don't need to change — the partition is applied in-memory during
  classification, not stored on the molecules.

  New tests: trimer classifier fixture verifying all three intermolecular
  modes (translate fragment A / B / C) come out labelled correctly, and
  intramolecular-stretch-and-bend of a mid-position fragment classified
  by frequency as before.

### Added

- **Commit-authorship policy + CI enforcement** (`CONTRIBUTING.md`,
  `.github/workflows/attribution.yaml`,
  `.github/scripts/check_ai_attribution.py`). The authors of a commit have
  to be people: AI assistants may be used freely and disclosed freely
  (`Made-with:` trailers, message-body mentions), but must not appear as
  commit author, committer, or `Co-authored-by:` trailer. A pull-request-only
  CI job checks the commits each PR adds (merged history is never
  retro-failed); detection keys on vendor email addresses, bot accounts, and
  assistant product names, never on a bare first name. Adapted from the
  equivalent policy in the stvogt/mace fork. BEEP's first CI workflow.

- **New workflow `be_assemble_periodic`** — extraction workflow closing the
  periodic BE pipeline. Reads the paired MACE-electronic + explicit-dispersion
  single-point energies submitted by `be_comp_periodic` (from the
  `<smol>_<slab>_be_sp`, `<smol>_<slab>_surface_be_sp`, and `<smol>_gas_be_sp`
  SinglepointDatasets), sums electronic + dispersion per record to composite
  BE-LOT totals, and computes per site
  `BE = E(complex) - E(bare_site) - E(adsorbate_gas)` shifted by a
  configurable `zpve_correction_kcal_mol` (scalar per adsorbate; kept as
  config so no periodic Hessians are ever needed). Only sites COMPLETE in
  all three record sets yield a BE; missing/errored records are logged and
  skipped. Writes `<molecule>/data/<prefix>_<slab>.csv` per slab (with
  intermediate hartree energies for auditability) and a
  `<prefix>_summary.csv` across all slabs.

- **New workflow `be_comp_periodic`** — submission workflow for periodic
  binding energies on `sampling_periodic` outputs. Range-separated: pairs
  an MLP electronic spec with an explicit dispersion spec, using the
  QCEngine MACE + `dftd3`/`dftd4` harness patches that read
  `cell`/`pbc` from spec keywords. Per slab, registers the two paired
  specs on:
  - `<smol>_<slab>_be_sp` — SPs on optimized complex geometries
  - `<smol>_<slab>_surface_be_sp` — SPs on per-site bare-surface geometries
  Once (shared across runs), same paired specs on `<smol>_gas_be_sp`
  (non-periodic — no cell/pbc in the gas-phase adsorbate SPs). Submits all,
  waits for completion. Only entries that appear COMPLETE in both the
  complex and bare-surface datasets from `sampling_periodic` are BE-
  evaluated, guaranteeing a per-site reference is always available for the
  downstream difference. Assembly of the per-site BE = E(complex) −
  E(bare_site) − E(adsorbate_gas), plus per-adsorbate ZPVE correction,
  happens in the paired `be_assemble_periodic` workflow. Config
  requires an MLP `be_electronic_lot` and a dispersion string
  (`mpwb1k-d4`, `b3lyp-d3bj`, etc.) — the dispersion program is inferred
  from the suffix by the existing `_split_dispersion` helper. New adapter
  helper `add_energy_spec` (sister of `add_gradient_spec`) added.

- **New workflow `sampling_periodic`** — grid-based binding-site sampling
  on periodic slabs, MLP-only (MACE). Complements the cluster-based
  `sampling` workflow. Single-pass optimization (no separate refinement
  stage): for each slab in a `surface_collection`, generates one adsorbate
  candidate per node on a PBC-aware xy grid, submits an MLP optimization
  per candidate, and reports per-slab unique binding sites via RMSD
  filtering.

  Ported from gbovolenta's `sampling_grid_noise_*.py` monoliths with the
  following changes to what shipped there:
  - **PBC-aware placement**: minimum-image nearest-atom search, wraps
    adsorbate atoms back into the cell after placement, and covers the
    full periodic footprint (no dead 1 Å borders — the monoliths' grid
    started at 1 Å from each edge to avoid boundary artefacts that PBC
    now handles correctly).
  - **Improved cavity strategy**: widened `cavity_z_scan_window_ang`
    (default 1.0 Å, was implicit 0.5 Å), scans the full z-range and
    picks the *best-fit* z rather than the first hit, and gracefully
    skips a grid node when no z qualifies (the monoliths would inherit
    a stale `z_shift` from a previous iteration).
  - **Sanity check over every adsorbate atom** (not just one), with an
    explicit `sanity_max_iter` cap on the retry loop so an impossible
    site is skipped cleanly instead of looping forever.
  - **Seeded RNG (`random_seed`)** for reproducibility.
  - **`freeze_below_z_ang` / `freeze_atoms`** — freeze bottom slab layers
    during optimization via geomeTRIC's `constraints` keyword (compressed
    to `$freeze / xyz i-j,k` ranges). Explicit atom list overrides the z
    threshold. Neither set = fully relaxed.
  - **Cell resolution**: config-level `cell` overrides; falls back to
    per-slab `surface.extras['cell']` if unset. `pbc` defaults to a 2D
    slab `[True, True, False]`.
  - **Adsorbate auto-centering in xy**: every generated candidate is slid
    (unconditional, unwarrantied) so the adsorbate COM sits at the periodic
    cell center. Pure PBC gauge shift — energy, gradient, Hessian invariant
    — but makes visualisation and human inspection dramatically cleaner
    (no boundary-straddling adsorbates). Z is left untouched so
    `freeze_below_z_ang` still picks the same atoms and the vacuum gap is
    preserved.
  - **Aggregate `all_sampled_sites_<slab>.xyz`** always written under
    `data/`, showing the slab plus every accepted adsorbate copy in its
    *pre-centering* placement — matching the monoliths'
    `all_sampled_sites.xyz` output and preserving spatial coverage for
    visual inspection. Per-candidate xyz files remain gated behind
    `store_initial_structures=true`.
  - **Per-site bare-surface companion opts** — for every unique confirmed
    binding site (after the RMSD filter), the adsorbate is stripped from
    the optimized complex and the remaining surface is re-optimized under
    the same LOT + freeze policy + cell/pbc. Results land in a sibling
    OptimizationDataset `<smol>_<slab>_surface` with entry names matching
    the sampling entries (1:1). Point: recover the site-specific slab
    deformation energy on amorphous ice — using a shared bare-slab
    reference for every site drops that ~5-20 meV contribution. Cheap
    with an MLP, prohibitive with DFT. Feeds the future
    `be_comp_periodic` / `be_assemble_periodic` workflows.

  Requires the QCEngine MACE-harness patch that reads
  `keywords['cell']` / `keywords['pbc']` from the QC specification; the
  cell + pbc are threaded through those keywords per submission so
  QCFractal caches remain correct (different cells hash to different
  specs). See `examples/sampling_periodic.json` and
  `beep --schema sampling_periodic` for all fields.

- **User-settable geomeTRIC keywords in `sampling`.** Both stages of the
  sampling workflow now expose the geomeTRIC `optimization_spec.keywords`
  dict to the config: `SamplingConfig.sampling_opt_keywords` and
  `SamplingConfig.refinement_opt_keywords`. Fully backward-compatible:
  both fields default to `None`, and the workflow's prior defaults are
  preserved unchanged — sampling stays at `{"maxiter": 125}` (geomeTRIC's
  own default `coordsys: tric` continues to apply), refinement stays at
  `keywords: None`. When set, sampling-stage keywords are *merged over*
  the built-in `{"maxiter": 125}` (user's `maxiter` wins if given, other
  keys added alongside); refinement-stage keywords replace the empty
  default. Motivating case: on large surfaces (W200 ≈ 600 atoms) the
  `tric` coordinate system incurs an O(N^2-N^3) Wilson/G-matrix cost per
  step and dominates wall time; setting `sampling_opt_keywords:
  {"coordsys": "cart"}` in the config now switches to Cartesian
  coordinates without a source edit.

### Changed

- **Adaptive surface-anchored binding-site sampler** (now the default;
  the legacy sampler selectable via the new config field
  `SamplingConfig.sampling_method: "adaptive" | "sphere"`, default
  `"adaptive"`). Replaces the spherical-shell scheme,
  which placed candidates on a single averaged-radius sphere and so floated over
  concavities and undersampled large clusters. The new sampler is
  composition-agnostic (per-element vdW radii only — water, CO2, methanol and
  mixed ices work with no O/H-bond assumptions), probes a fan of directions per
  accessible surface atom and seats each candidate at vdW *contact* with the
  surface (nearest-atom ≈ contact distance) rather than floating it a fixed gap
  above, then ranks placements by coordination so multi-atom hollow/bridge sites
  are kept preferentially over single-atom caps — coverage is curvature-unbiased
  and reaches into pockets, and it runs unchanged from a water trimer to W200.
  `sampling_condition`
  now sets thoroughness as coverage → orientations → off-atom jitter
  (`sparse` = 1/6 of anchors, `normal` = 1/2, `fine` = all, `hyperfine` = all +
  2 orientations + ~1 Å lateral jitter into hollow/bridge sites) instead of the
  redundant multi-shell height passes, and `max_structures` scales with the
  accessible-anchor count rather than the old `n_water//3` (W200: ~371 vs 66).

### Fixed

- **`nm_sampling`: `NameError` on the imaginary-mode abort path** — the abort
  and pre-run messages referenced an undefined `molden_dir`; they now point to
  `res_folder/normal_modes_<system>.molden` (where the molden files are
  actually written), so an equilibrium geometry with imaginary frequencies
  aborts with a useful message instead of crashing.

- **`nm_sampling`: transient gradient failures no longer drop a functional
  from the report** — `wait_for_nm_completion` now auto-resets errored leaf
  records up to `max_resets` (default 2) times each before finishing, so
  transient infrastructure failures (ManagerLost / worker walltime, common on
  clusters like Aire) recover and the affected functional is kept. Genuine
  failures (e.g. SCF non-convergence) exhaust their retries and are left in
  ERROR, so the poll still terminates.

## [0.15.0] — 2026-07-11

### Added

- **Range-separated MACE binding energies (`mace_dispersion`) in `be_hess`.**
  A MACE model trained on the *electronic* (dispersion-subtracted) energy can
  now be paired with an analytic dispersion tail so `be_hess` reproduces the
  full method — mirroring how DFT-D is stored as separated
  (bare-functional + bare-dispersion) specs. New optional
  `BeHessConfig.mace_dispersion` field (e.g. `"mpwb1k-d4"`, `"mpwb1k-d3bj"`);
  when set, `compute_be_mace_energies` submits, per model and stoichiometry, an
  additional `dftd4`/`s-dftd3` spec named `"<model-stem><suffix>"` with the
  functional params taken from the prefix (`mpwb1k`), and
  `fetch_reaction_values` sums the electronic MLP column with its dispersion
  into the composite `"<model-stem>-d4"` BE (extended to pair basis-free MLP
  electronic columns, reusing the DFT-D `_split_dispersion` /
  `DISPERSION_PROGRAMS` single source of truth). This lets a short-range MLP
  handle the electronic PES (cutoff-adequate) while the exact long-range
  dispersion — untruncated and transferable to periodic surfaces — is added
  analytically; the dispersion records persist on the server via the standard
  `dftd4`/`s-dftd3` harnesses. A companion `BeHessConfig.dispersion_tag`
  (defaults to `energy_tag`) routes the CPU-only dispersion single-points to a
  separate queue, so the electronic MLP energies can run on a GPU manager while
  `dftd4`/`s-dftd3` run on a CPU manager.

- **Atomic (single-atom) adsorbate support in `geom_benchmark`.** The workflow
  now recognizes an atomic adsorbate the same way `sampling` and `be_hess` do:
  when the adsorbate is absent from the `small_molecule_collection`
  OptimizationDataset it is fetched from the `atoms_collection`
  SinglepointDataset (new `GeomBenchmarkConfig.atoms_collection` field, default
  `"atoms"`). Because a lone atom has 0 internal DOF (geomeTRIC rejects
  <2-atom inputs) and has no geometry to compare, the atom monomer is excluded
  from the optimize+RMSD structure set — only the surfaces and the complexes
  are benchmarked. Previously the forced monomer optimization errored for every
  functional, and `compare_rmsd` excludes any functional that errors on any
  structure, so the erroring monomer silently poisoned the entire ranking
  (empty result). Multiplicity is read from the atom's stored molecule.

- **Atomic (single-atom) adsorbate support in `energy_benchmark`.** The BE
  benchmark now handles an atomic adsorbate, mirroring `be_hess`: when the
  adsorbate is absent from the `small_molecule_collection` OptimizationDataset
  it is fetched from the `atoms_collection` SinglepointDataset (new
  `EnergyBenchmarkConfig.atoms_collection` field, default `"atoms"`). Unlike
  `geom_benchmark` the atom cannot be excluded — its single-point energy is
  required for the binding-energy stoichiometry (`BE = complex − surface −
  atom`) and for the CCSD(T)/CBS reference — so the atom molecule is used
  directly as the small-molecule fragment (no optimized geometry needed) in
  both the reference-geometry fragment set and `_fetch_be_molecules`. The
  open-shell CCSD(T)/CBS keywords (`reference uhf`, `qc_module OCC`) are already
  selected automatically from the adsorbate multiplicity.

- **MACE machine-learning potential support (`mace_model`) in `sampling`,
  `be_hess`, and `extract`.** A `LevelOfTheory` can now point at a serialized
  MACE model file via `mace_model`, which mutes `method`/`basis`/`program`:
  the spec runs through the stock QCEngine `mace` harness (`program: mace`,
  method = model file path, no basis), and the model file stem (e.g.
  `mace-polar-ft0` for `.../mace-polar-ft0.model`) names the spec, reaction
  datasets, dataframe columns, and output files. Committees are one spec per
  member model on the same datasets.
  - `BeHessConfig.mace_models` submits MACE BE single points via the new
    adapter helper `compute_be_mace_energies`. The `bsse` (counterpoise)
    stoichiometry is skipped for MACE specs — MLPs carry no basis functions
    and ghost atoms would be treated as real atoms; extract MACE binding
    energies with `stoichiometry: be_nocp`, `de`, or `ie`.
  - Constraints (validated): model file paths must be all-lowercase (qcportal
    lowercases QCSpec methods server-side) and the file stem must not contain
    underscores (LOT names split on `_`).
  - `extract` keeps basis-less BE columns (MACE aliases, tight-binding like
    `gfn2-xtb`); only bare-dispersion bookkeeping columns are dropped.
    `be_methods` entries may name basis-less methods directly.
  - Requires `mace-torch` in the compute-manager worker environment; the
    QCEngine MACE harness needs the `Configuration` compatibility fix for
    `mace >= 0.3.10` (patched on the `nothung-deploy` QCEngine fork).

- **Many-Body Expansion binding-energy workflows (`mbe` / `mbe_extract`),
  ported from the standalone `beep-mbe` package (v0.1.0 @ `44a90e6`).** These
  provide an alternative route to binding energies on the *same* binding sites
  produced by `sampling` / `be_hess`, re-evaluated at a (typically higher)
  level of theory via n-body fragmentation on a qcmanybody `ManybodyDataset`
  plus a monomer `SinglepointDataset`.
  - `mbe` submits and (optionally) monitors the many-body computations.
  - `mbe_extract` assembles per-site binding energies (`be_data/total_be.csv`)
    plus n-body decomposition tables (`decomp__<spec>.csv`,
    `contrib__<spec>.csv`) and a text report.
  - `mbe_extract` can optionally apply a **read-only** ZPVE correction borrowed
    from a prior `be_hess` run on the same sites (`total_be_zpve.csv`); it never
    submits or mutates the `be_hess` datasets, and sites without a usable
    Hessian are reported as `NaN`.
  - New adapter helpers (`get_or_create_manybody_dataset`,
    `mbe_levels_to_qc_specifications`, `build_manybody_specification`,
    `wait_for_manybody_completion`, `wait_for_dataset_records`) are strictly
    additive; the existing `wait_for_completion` used by `sampling` /
    `geom_benchmark` is unchanged.
  - Config uses BEEP's Pydantic-v2 schema (nested `server.*`, object-style
    `levels`); the standalone package's flat JSON format is not supported.
    See `examples/mbe.json` and `examples/mbe_extract.json`.
  - `pandas` is now an explicit runtime dependency.

- **MBE truncation-error estimate in `mbe_extract`.** Each site/spec now gets a
  symmetric error bar on the binding energy from a geometric extrapolation of
  the uncomputed n-body tail (`bar = |Δn|·r/(1−r)`, `r = |Δn/Δn−1|`), written to
  `be_data/convergence__<spec>.csv` and rendered as `BE ± bar (converged?)` in
  the report. The bar is a magnitude only — no signed `BE_∞` — because the sign
  of the next term is not predictable. 2-body-only runs report `n/a` (no
  convergence information); a non-shrinking series is flagged not converged. The
  `converged` flag uses a configurable `convergence_tol` (default 0.05). The
  existing `total_be` / `decomp` / `contrib` CSVs are unchanged.
- **ZPVE correction is now a friction-free toggle.** `mbe_extract`'s
  `zpve.hessian_clusters` is optional; when omitted, the `be_<MOL>_*` datasets
  are auto-discovered from the server (still strictly read-only). Turning ZPVE
  on is just `"zpve": {"enabled": true}`.
- **MBE level validation.** `mbe` now requires contiguous body-order indices
  (`1..N`, no gaps), matching qcmanybody's expectation. Per-tier levels of
  theory (a distinct method/basis per body order) and per-site selection
  (`entries`, omit for all sites of the cluster) are documented in the examples.

- **Multi-program Hessian submissions in `be_hess` (ORCA, Gaussian).** New
  `hessian_method_and_keywords()` helper in the adapter routes Hessian
  submissions through a per-program dispersion-keyword table
  (`DISPERSION_KEYWORD_SPEC`), so DFT-D methods work on ORCA and Gaussian in
  addition to psi4. The compound method (e.g. `b3lyp-d4`) is split into the
  bare functional plus the program's native dispersion keyword on the
  harness's escape-hatch field:
  - **ORCA**: `simple_input` = `D4` / `D3BJ` / `D3ZERO` for `-d4` / `-d3bj` /
    `-d3`. Raises on `-d3m` / `-d3mbj` (no native ORCA keyword).
  - **Gaussian**: `route_input` = `EmpiricalDispersion=GD3BJ` /
    `EmpiricalDispersion=GD3` for `-d3bj` / `-d3`. Raises on `-d4` /
    `-d3m` / `-d3mbj` (not in Gaussian).

  psi4-style keywords (`dertype`, `reference`) are dropped for the ORCA and
  Gaussian paths, since both programs' Hessians are analytic and UKS/UHF
  follows from the molecule multiplicity. The Hessian physics matches the
  psi4 workflow: the same Grimme library supplies the dispersion second
  derivatives, invoked by the target program. Only the BE-*energy* stage
  keeps dispersion in separate `dftd3` / `dftd4` records; the psi4 path is
  unchanged.

### Fixed

- Basis-less optimization LOTs (e.g. `gfn2-xtb`, MACE aliases) no longer
  crash `sampling` (refinement LOT-name concatenation) or `be_hess`
  (`opt_level_of_theory` method/basis split); reaction-dataset names now
  omit the basis segment for basis-less LOTs.

## [0.14.0] — 2026-07-07

### Added

- **Raw force-delta arrays persisted to disk** for both `nm_sampling` and
  `geom_benchmark` trajectory analysis. The per-functional flat arrays of
  per-Cartesian-component force deltas (and per-atom energy deltas) are
  now saved alongside the metrics JSON as a single NumPy `.npz` file,
  loadable with `np.load(path)` → `{functional_name → array}`. Lets
  users plot their own per-functional histograms of `(F_DFT - F_ref)`
  without re-running the workflow.

- **Atom sampling and atom be_hess no longer crash on missing entries.**
  `qcportal>=0.63` raises `PortalRequestError` (HTTP 400 "Missing N
  entries: …") — not `KeyError` — when an entry name isn't in the
  dataset. Both `sampling.py` and `be_hess.py` caught only `KeyError`
  on the OptimizationDataset lookup before falling through to the
  atoms_collection, so single-atom adsorbates (e.g. C, N, O) crashed
  the workflow instead of being routed to the SinglepointDataset
  fallback. Fixed at the adapter layer: `fetch_opt_record` and
  `fetch_atom_molecule` now translate the specific "Missing entries"
  PortalRequestError into `KeyError` while letting genuine
  server/network/auth failures propagate, so the workflow's
  `except KeyError` does the right thing and unrelated transient
  failures aren't silently misrouted as "atom not found". Five
  regression tests added.

- **Clearer per-LOT submission log in `be_hess` / `energy_benchmark`.**
  `compute_be_dft_energies` used to log `Existing N  Submitted M` per
  LOT, where both `N` and `M` come from `ds.submit()`'s
  `InsertCountsMetadata` — *newly* linked vs *newly* submitted in this
  call. On re-runs where every reaction was already linked to the
  dataset from a prior run, both would correctly be 0, but
  "Existing 0  Submitted 0" reads as "nothing on the server" rather
  than "nothing changed." The line now distinguishes the two cases:
  if both metrics are 0, it logs "all reactions already linked to the
  dataset (no new submissions)"; otherwise it logs newly-submitted +
  newly-linked counts explicitly.

- **Refinement summary table in `sampling` workflow.** After refinement
  polling finishes, the workflow now logs a per-cluster breakdown of
  COMPLETE / ERROR / TOTAL refinement opts in the same layout as the
  sampling summary.

- **Functional averaging and over/under-bind tag in `energy_benchmark`.**
  Two related additions to the BE per-group MAE table:

  - New `functional_averages: List[List[str]]` config field. Each group
    is a list of `method_basis` LOT strings. For every group, the
    workflow computes the mean BE per binding site across the listed
    functionals and reports it as one additional row labelled
    `DFT_Average_<N>` under a dedicated "Averages" group in the per-group
    output. Members not present in the benchmark set are dropped with a
    warning. Empty default; configs without the field work unchanged.

  - The per-group MAE table now carries a `Bias` column and a
    human-readable tag indicating whether each functional (or DFT
    average) over- or under-binds the reference. Sign convention:
    negative bias → overbinds, positive bias → underbinds. Tag
    thresholds (kcal/mol): `|bias| < 0.05` → balanced; `< 0.5` → mild
    over/underbind; else → strong over/underbind. Surfaces in the
    standard BE table and the gCP-corrected BE table when
    `gcp_correction=true`.

- **Errored-record reporting at the end of each `be_hess` section.**
  After each completion check (BE pass 1, BE pass 2, Hessian),
  `be_hess` now logs the record IDs that ended in ERROR as a single
  space-separated line, making them easy to copy-paste into a reset
  command. No log noise when there are no errors. Driven by a new
  `report_errored_records(client, ids, logger, section_name)` helper
  in the adapter, reusable by any workflow that wants the same
  end-of-section error summary.

### Fixed

- **Trajectory + SP gradient lookup falls back to `return_result`.**
  `fetch_sp_energy_gradient` and `get_optimization_trajectory` in the
  adapter previously read only `properties.return_gradient`. That field
  is optional per the qcelemental spec: psi4 populates it, but molpro
  (used for the F12 reference geometries in `geom_benchmark` /
  `nm_sampling`) leaves it unset and instead writes the gradient to
  `return_result` on gradient-driver records. The reader was silently
  returning `None` for the gradient on those records, so the
  trajectory force-RMSE / nm_sampling force-RMSE analyses on F12-stack
  data were collapsing to no-data and producing silent zeros. Both
  helpers now try `properties.return_gradient` first and fall back to
  `return_result` when the spec driver is gradient.

- **`pre_exp` moments of inertia now COM-shifted.**
  `get_moments_of_inertia` in `core/pre_exponential.py` evaluated the
  inertia tensor in the input frame without first translating to the
  centre of mass, so any COM offset in the geometry inflated all three
  eigenvalues via the parallel-axis theorem. Compounding this,
  `workflows/pre_exp.py` was passing the raw, un-aligned coordinates
  to `get_moments_of_inertia` while computing (and then discarding) the
  output of `align_to_z_axis`. Drift observed: ~18% on H2O,
  <1% on NH3/CH3OH/CH3CN where the optimiser happened to land near
  origin, ~0 on linear molecules where `Ia=0` is preserved. Two
  coordinated fixes:
  1. `get_moments_of_inertia` now COM-shifts unconditionally before
     building the tensor, so the function is correct regardless of
     what the caller passes.
  2. The `pre_exp` workflow now passes the aligned coordinates (the
     output of `align_to_z_axis`) into `get_moments_of_inertia`, so
     the principal-axis frame is used downstream rather than
     computed-and-discarded.

- **`compute_hessian` reuses existing Hessian records regardless of
  spec keywords.** The DB now holds a mix of migrated Hessians (stored
  with empty `keywords={}`) and newly-submitted ones (stored with
  `keywords={"function_kwargs": {"dertype": 1}}`), and
  `add_singlepoints`' server-side `find_existing` matches strictly on
  the keyword dict. Migrated records were therefore missed and
  re-submitted. `compute_hessian` now does a keyword-agnostic
  pre-query on `(molecule_id, method, basis, driver=hessian, status=complete)`
  before calling `add_singlepoints`, treats every matching record as
  reusable, and submits new Hessians only for the molecules without a
  hit. Eliminates redundant — and individually expensive — Hessian
  re-computes on migrated species. The 2022-era keyword on new
  submissions is unchanged, so future-pure-v0.14 datasets keep the
  full spec.

- **`be_hess` and `energy_benchmark` no longer wipe existing reaction
  datasets on every run.** `create_or_load_reaction_dataset` and
  `create_reaction_dataset` were unconditionally deleting and
  recreating each per-stoichiometry ReactionDataset (`_bsse`,
  `_be_nocp`, `_ie`, `_de`) on every workflow launch, which wiped
  every spec/entry registration the dataset had accumulated from
  previous runs at other LOTs. Underlying singlepoint records
  survived (`delete_records=False`) but were left orphaned. Both
  helpers are now true get-or-create: the dataset is returned
  unchanged if it exists, only created when missing. qcportal 0.63+
  `add_specification` and `add_entry` are already idempotent on the
  dataset, so a second `be_hess`/`energy_benchmark` run at a new LOT
  now layers its specs on top of the existing dataset instead of
  destroying prior LOT data.

### Changed

- **Reverted unconditional SCF-accelerator override in `be_hess`.**
  0.13.0 unconditionally injected `scf_initial_accelerator: NONE` into
  the BE single-point keywords to bypass Psi4's default ADIIS warmup.
  That changed the QCSpecification hash and broke `find_existing` cache
  hits against records on the server that were submitted without the
  keyword. 0.14.0 reverts to the pre-fix behaviour: no
  scf_initial_accelerator override is sent, Psi4's default SCF
  acceleration is used, and existing records continue to match. Users
  who hit ADIIS-induced SCF convergence failures should handle them
  per-record (e.g. reset the failed records) rather than via a
  workflow-wide keyword.

- **`sampling` workflow log order.** The `SAMPLING SUMMARY` table (binding
  sites per cluster) now appears immediately after the sampling loop
  finishes and *before* the refinement-polling block, instead of after.
  Makes the sampling vs refinement stages distinguishable in the log;
  pairs with the new refinement summary table at the very end.

- **Plot generation now opt-in across all workflows.** `geom_benchmark`
  and `energy_benchmark` (alongside `extract`, which was already opt-in)
  now expose a `generate_plots: bool` config field defaulting to
  `False`. JSON results are always written; SVG plots only when the
  flag is set. Set `"generate_plots": true` in the workflow JSON to
  restore the previous always-plot behavior.

- **Uniform output directory layout across every workflow.** All
  workflows now write to `<cwd>/<molecule>/` with a single shape:
  ```
  <molecule>/
  ├── <workflow>_<molecule>.log         (workflow log)
  ├── <workflow>_<molecule>.json        (copy of input config)
  └── data/                             (only if the workflow produces outputs)
      ├── <misc files>                  (CSV / NPZ / xyz / molden / dat)
      ├── json/                         (all JSON outputs)
      └── plots/                        (all SVG / PNG plots)
  ```
  Replaces the per-workflow `<molecule>/<workflow>/` nesting and the
  ad-hoc layouts that previously varied between workflows. Downstream
  consumers that hardcode the old paths (e.g. notebooks reading
  `<molecule>/<workflow>/json_data/...`) need updating to the new
  `<molecule>/data/json/...` form.

## [0.13.0] — 2026-06-08

### Added

- **`nm_sampling` workflow** — per-functional force-RMSE benchmark on
  normal-mode-displaced geometries. For each binding site, BEEP
  computes a cheap Hessian (default `hf/def2-svp`, configurable),
  diagonalises it via Psi4/qcelemental, classifies every vibrational
  mode as **intermolecular / bending / stretching** by
  fragment-centre-of-mass projection, picks the lowest-frequency modes
  in each band up to per-band caps (defaults `3 / 2 / 1`), and
  generates ± displaced geometries at per-band RMS Cartesian
  amplitudes (defaults `0.08 / 0.05 / 0.03 Å`). At every displacement
  it submits a CCSD(T)/aug-cc-pvtz reference gradient plus a DFT
  gradient per functional in the geom_benchmark pool, then reports
  **per-Cartesian-component force RMSE** vs CCSD(T) grouped by
  functional category. Same per-group log layout as the
  `geom_benchmark` trajectory output. Invoke via `beep --config
  nm_sampling.json`; `beep --schema nm_sampling` for the full config.

  *Why force RMSE on displacements and not just along the trajectory.*
  Trajectory sampling probes a one-dimensional path; normal-mode
  displacements span the soft, chemically-relevant degrees of freedom
  off equilibrium. R2SCAN-3c wins on the H2/W1 sanity benchmark at
  18.5 meV/Å; HF-3c last at 805 meV/Å — both consistent with
  expectation. Single-metric ranking (no z-score machinery) — the
  per-group summary picks winners by raw RMSE.

- **Trajectory analysis in `geom_benchmark`** (default-on). For each
  DFT functional, BEEP now submits SP + gradient calculations at every
  geometry along the reference optimization trajectory and reports the
  per-Cartesian-component **RMSD of the force** (meV/Å) vs the
  reference. Combined with the existing equilibrium-geometry RMSD via
  a z-score-weighted ranking (weights configurable via
  `score_weights`). Same per-group table layout as the existing
  `BENCHMARK RESULTS` section; appears immediately below it in the
  workflow log. Inspired by the MLP test-set validation in Bovolenta
  et al. 2025 (A&A, in press; arXiv 2508.14219), Appendix C.2 /
  Fig C.1 / Table C.1. Set `trajectory_analysis: false` in the
  workflow config to keep the legacy eq-geometry-only behaviour.

  *Why force RMSD and not energy MAE/RMSE.* Absolute total energies
  carry method-specific offsets (correlation, basis, BSSE) that
  dominate any cross-method comparison and aren't relevant to
  geometry quality. Gradient deviations reflect PES shape, which is
  what geometry optimization actually cares about. For relative-energy
  comparison use the `energy_benchmark` workflow. RMSD (rather than
  MAE) is used to penalise occasional large failures — a single bad
  gradient step is exactly the failure mode that derails a real
  geometry optimization.

- **`combined_zscore_ranking` is weight-driven.** The metrics combined
  are taken from `weights.keys()`, so the same function serves the
  2-metric geom-benchmark case (`rmsd_eq` + `rmsd_force`) and any
  future N-metric variant without modification.

### Changed

- `geom_benchmark` now performs additional SP + gradient submissions
  per functional × reference-trajectory step when
  `trajectory_analysis` is enabled (the new default). Existing 0.12.x
  configs that don't set the field will pick up this behaviour
  automatically; opt out with `trajectory_analysis: false`.

- `GeomBenchmarkConfig.score_weights` now defaults to
  `{"rmsd_eq": 1.0, "rmsd_force": 1.0}` (equal weighting of the two
  metrics that survive the geom-benchmark physical filter).

### Fixed

- `be_hess`: bypass Psi4's default ADIIS warmup by setting
  `scf_initial_accelerator: NONE` in the SCF keywords. On Psi4 1.10
  the ADIIS path triggered intermittent SCF convergence failures on a
  fraction of BE records. Verified on H2O / B3LYP / def2-SVP that pure
  DIIS reaches the same converged energy (Δ < 1e-13 Ha) without the
  ADIIS markers in the Psi4 output. No effect on already-submitted
  records — only new submissions pick up the override.

## [0.12.0] — 2026-05-30

The first BEEP release in the qcportal 0.63+ era. The QCFractal stack
underwent a major rewrite between 0.15 and 0.50+, and this release
catches BEEP up — workflows, dataset I/O, spec representation, and CLI
all have new shapes. **This is a breaking release**: existing 0.6.x
configs and the legacy `launch_*.py` scripts no longer work.

### Breaking changes

- **Requires qcportal ≥ 0.63 and a QCFractal server ≥ 0.63.** Old
  qcportal-0.15 installations are no longer supported.
- **Single CLI entry point.** The standalone `launch_sampling.py`,
  `launch_energy.py`, `launch_be_hess.py`, and `launch_pre_exp.py`
  scripts have been replaced by a single command:
  ```
  beep --config <input.json>
  ```
  Workflows are selected by the `"workflow"` field in the JSON
  configuration. See `beep --workflows` to list available workflows and
  `beep --schema <workflow>` to dump a workflow's full field schema.
- **JSON-driven configuration with Pydantic validation.** Each workflow
  now has a Pydantic `*Config` model in `beep/models/` that validates
  the input JSON before execution. Type errors surface at parse time
  rather than mid-run.
- **`ReactionDataset` split by stoichiometry.** qcportal 0.63's
  `ReactionDataset` can only hold one reaction definition, so BEEP
  creates **four separate datasets per cluster**:
  `be_<MOLECULE>_<CLUSTER>_<METHOD>_<BASIS>_<stoich>` where `<stoich>`
  is one of `bsse`, `be_nocp`, `ie`, or `de`. Old 0.15-style datasets
  with multiple stoichs in one collection are no longer compatible.
- **Separated-pair dispersion specs.** Dispersion-corrected DFT
  (e.g. `b3lyp-d3bj/def2-tzvpd`) is now stored as a **separated pair**
  — a bare DFT spec on `psi4` plus a bare dispersion spec on `dftd3`
  or `dftd4` — matching the qcportal 0.63 migration layout. The
  recombined composite column is computed at fetch time. Integrated
  specs (method-with-dispersion-suffix carrying a basis) are skipped
  with a warning.
- **Lowercase spec convention.** All QCSpec names (`method`, `basis`,
  level-of-theory strings) are now normalized to lowercase at the
  model layer. Dataset names remain uppercase (BEEP convention).
- **Python ≥ 3.10 required.** Previous releases supported 3.8.
- `stoich="default"` / `stoich="int"` (qcportal 0.15 aliases) are
  gone — use `"bsse"` and `"ie"` respectively.

### Added

- **Conservative auto-recovery in `check_jobs_status`.** Parent
  `ReactionRecord` services stuck at ERROR are auto-reset *only when
  `record.children_errors` is empty*, restoring the qcportal-0.15 UX
  where resetting an errored child externally was enough to make the
  workflow continue without a be_hess restart. Real chemistry-error
  services are left alone for human inspection. Opt out via
  `auto_recover_services=False`.
- **Refinement-completion monitor in `sampling` workflow.** After the
  per-cluster sampling loop, the workflow polls all refinement-LOT
  optimizations and exits only when they reach terminal state. Max
  wait one week.
- **`fetch_reaction_values` helper** in
  `beep/adapters/qcfractal_adapter.py`. Returns a DataFrame indexed by
  entry name with composite columns built from separated-pair specs.
- **New tutorial notebook**:
  `tutorial/data_generation/05_restart_errored_reactions.ipynb` —
  walks through identifying and recovering from infrastructure vs
  chemistry errors across all four stoichiometry datasets, with regex
  classification.
- **Regression test suite** (`tests/`) — 255 tests covering the
  qcportal-0.63 adapter, workflows, model validation, and the new
  recovery / restart logic.
- `examples/` directory with template JSON configs for each workflow.
- `beep --workflows` lists available workflow names.
- `beep --schema <workflow>` dumps the full Pydantic schema (every
  field, type, default, description).
- `networkx` added as a direct runtime dependency (qcelemental's
  `align(atoms_map=True)` path, used by BEEP's symmetric RMSD filter,
  requires it but doesn't declare it itself).
- `pyproject.toml` packaging; `setup.py`/`setup.cfg` deprecated.

### Changed

- **All workflows refactored** to read a single JSON config:
  `sampling`, `be_hess`, `extract`, `pre_exp`, `geom_benchmark`,
  `energy_benchmark`. QCFractal I/O isolated into
  `beep/adapters/qcfractal_adapter.py`; pure computation logic moved
  to `beep/core/`.
- `fetch_reaction_values` column labels are lowercase (matching the
  lowercase spec convention).
- Tutorial notebooks 01-04 updated to the new qcportal 0.63 API and
  the `beep --config` CLI. Markdown text rewritten throughout (typos
  fixed; legacy CLI flags removed; per-stoich `ReactionDataset` split
  documented).
- Docs (`docs/index.rst`, `docs/getting_started.rst`) updated for
  qcportal 0.63+ and Python 3.10+. `readthedocs.yml` migrated to RTD
  v2 schema (ubuntu-24.04, python 3.10, pip-based install).

### Fixed

- **Sampling re-run at a new LOT.** Pre-existing entries are now
  submitted at the new sampling LOT (server-side dedupe handles the
  same-LOT case as a no-op). Previously the workflow short-circuited
  on "all entries already exist" and submitted zero new opts.
- **Atom-fragment ZPVE extraction.** Single-atom fragments
  (e.g. `C`, `N`, `O` on water clusters) no longer crash
  `zpve_correction` with `RuntimeError: Missing hessian`. The fix
  distinguishes `None` (genuinely missing) from `0.0` (valid atomic
  ZPVE).
- **`check_jobs_status` infinite-loop.** A transient HTTP 500 on
  `reset_records` no longer wedges the polling loop. Reset is
  best-effort: each service ID is reset at most once per call
  regardless of success/failure.
- **`get_zpve_mol` skips incomplete hessian records.** A "complete"-
  tagged hessian record whose `properties` is `None` (server
  inconsistency, mid-write state) no longer crashes downstream with
  `AttributeError: 'NoneType' object has no attribute 'get'`. The
  function now queries with `status=RecordStatusEnum.complete` and
  defensively filters records with null properties.
- **`get_job_ids` handles missing entries.** The `sampling` workflow
  no longer crashes with `PortalRequestError: Missing N entries` when
  pose generation returns fewer structures than requested.
- **be_hess parent-service ERROR recovery.** Workflow no longer logs
  "Some jobs have ERROR" indefinitely after the polling loop exits;
  conservative auto-recovery transitions stale-state parents back to
  COMPLETE when their children are clean.
- **`fetch_reaction_values` integrated-spec handling.** Integrated
  dispersion specs (e.g. `pbe-d3bj_def2-tzvp` as a single spec) are
  skipped with a warning rather than producing duplicate columns.
  BEEP no longer supports the integrated layout — only the
  separated-pair form matching the qcportal 0.63 migration.
- Stoichiometry-type lookup is consistent across `fetch_reaction_values`
  consumers; `extract` workflow's bare-DFT column drop heuristic uses
  lowercase suffixes.
- Pre-exp / extract / be_hess level-of-theory case mismatch (validators
  upper-cased what the server stored lowercase) — all flipped to
  lowercase.

### Removed

- `launch_sampling.py`, `launch_energy.py`, `launch_be_hess.py`,
  `launch_pre_exp.py` standalone CLI scripts — replaced by the unified
  `beep --config <input.json>` CLI.
- `setup.py` (replaced by `pyproject.toml`).
- Legacy U-prefix unrestricted-spec handling in `geom_benchmark` —
  unrestricted treatment now comes via keywords or multiplicity.
- Dead `query_results` adapter helper.

### Notes for users upgrading from 0.6.x

1. **Server side**: ensure your QCFractal server is on the 0.63+ stack.
   Data from older qcportal-0.15 servers needs to be migrated (a
   migration path exists for ReactionDatasets — see the `migration/`
   directory if present).
2. **Client side**: install the `beep-0.12` conda environment from the
   `qcfractal-0.64` clone (see README) and `pip install .` on top.
3. **Config files**: convert your old argparse-based command lines to
   the new JSON format. `beep --schema <workflow>` is the easiest way
   to generate a starting template; the `examples/` directory has one
   per workflow.
4. **Existing datasets**: ReactionDatasets created by BEEP 0.6.x cannot
   be read directly. The data on the v0.63 server should already be in
   the four-stoich layout (one dataset per stoichiometry type) if the
   migration script was used.

[0.12.0]: https://github.com/QCMM/beep/releases/tag/v0.12.0

## Periodic workflow migration

Periodic sampling, BE/IE/DE computation and assembly, their configuration models,
geometry helpers and examples moved to beep-mlp. Use
`beep-mlp <workflow> --config FILE` with the existing JSON configuration.
BEEP retains cluster MACE support and shared QCFractal utilities.
