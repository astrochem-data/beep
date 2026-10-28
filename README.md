![logo_BEEP](https://user-images.githubusercontent.com/7481702/178007641-1f4260b3-dd34-4e39-9a51-286b076c5ea8.png)

# BEEP — Binding Energy Evaluation Platform

BEEP is a binding energy evaluation platform and database for molecules on interstellar ice-grain mantles.
It automates the full computational pipeline, from structure sampling through geometry optimization, binding energy computation, and data extraction. It is build 
using a [QCFractal](https://github.com/MolSSI/QCFractal) (v0.63+) server as the computation backend.

## Installation

BEEP requires Python 3.10+ and the QCFractal v0.63+ stack. The supported install
path is to clone a `qcfractal-0.64` conda environment and pip-install BEEP on top:

```bash
conda create --clone qcfractal-0.64 -p /path/to/envs/beep-0.16
/path/to/envs/beep-0.16/bin/pip install .
```

Pure `pip install .` also works in any environment that already has the
QCFractal stack (qcportal, qcelemental, qcfractalcompute) available.

Direct dependencies declared in `pyproject.toml`:
`qcportal>=0.63`, `qcelemental`, `matplotlib`, `scipy`, `seaborn`.
`numpy`, `pandas`, `pydantic`, `tqdm`, and `pyyaml` come transitively through
`qcportal`.

**Note:** To run computations you also need a running QCFractal server (v0.63+)
with compute workers. See the [QCFractal documentation](https://docs.qcarchive.molssi.org)
for server setup.

## Usage

All workflows are driven through a single CLI entry point with a JSON configuration file:

```bash
beep --config input.json
```

The JSON file must contain a `"workflow"` key that selects which workflow to run.

### Discovering workflows and configuration options

List all available workflows:

```bash
beep --workflows
```

Dump the full JSON schema for any workflow (shows every field, its type, default value, and description):

```bash
beep --schema sampling
```

This is the easiest way to create a new configuration file — start from the schema and fill in the required fields.

## Workflows

| Workflow | Description |
|----------|-------------|
| `sampling` | Generate binding-site candidates on water clusters (adaptive surface-anchored or spherical placement), RMSD-filter them and refine the geometries |
| `be_hess` | Submit binding energy (with BSSE) and Hessian computations for the optimized cluster structures |
| `extract` | Extract binding energies, apply ZPVE corrections, and generate summary tables and plots |
| `mbe` | Submit and monitor many-body-expansion binding energies on existing binding sites at a higher level of theory |
| `mbe_extract` | Assemble MBE binding energies and n-body decomposition tables, optionally borrowing a ZPVE correction from `be_hess` |
| `pre_exp` | Compute transition-state-theory pre-exponential desorption factors over a temperature range |
| `geom_benchmark` | Benchmark DFT geometry optimizations and per-step force RMSD along the reference trajectory against high-level reference data |
| `energy_benchmark` | Benchmark DFT binding energies against high-level reference values (CBS extrapolation), with optional gCP correction |
| `nm_sampling` | Benchmark DFT gradients on ± normal-mode displacements against a high-level (CCSD(T)) reference, with per-band force-RMSE ranking |

Each workflow has a corresponding example configuration in the [`examples/`](examples/) directory:

```
examples/
  sampling.json
  be_hess.json
  extract.json
  mbe.json
  mbe_extract.json
  pre_exp.json
  geom_benchmark.json
  energy_benchmark.json
  nm_sampling.json
```

Periodic MLP workflows now live in **beep-mlp**. Existing periodic JSON files can be
run with `beep-mlp sampling_periodic --config input.json` (likewise
`be_comp_periodic` and `be_assemble_periodic`). Cluster workflows continue to
support MACE models through `mace_model`.

## Output

Each workflow creates an organized output directory under the molecule name:

```
N2/
  sampling/       # binding site structures, site_finder debug data, log
  be_hess/        # log, config copy
  extract/        # CSV tables, SVG plots, log
```

Output folders contain a log file and a copy of the input configuration for reproducibility.

## Tutorials

The [`tutorial/`](tutorial/) directory contains Jupyter notebooks for:

- **Data query** — how to query the BEEP database using QCPortal
- **Data generation** — step-by-step walkthrough of the full BEEP protocol

## Running tests

```bash
pip install .[test]
pytest
```

Have fun BEEPing!

### Copyright

Copyright (c) 2022, Stefan Vogt-Geisse, Giulia M. Bovolenta

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
