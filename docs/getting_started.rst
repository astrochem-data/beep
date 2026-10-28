Getting Started
===============

Prerequisites
-------------

- Python 3.10 or later
- A running `QCFractal <https://docs.qcarchive.molssi.org>`_ (v0.63 or later) server with compute workers (required for running computations; not needed for data extraction or analysis)

Installation
------------

Clone the repository and install with pip:

.. code-block:: bash

   git clone https://github.com/QCMM/beep.git
   cd beep
   pip install .

This pulls in the core dependencies: ``qcportal``, ``qcelemental``, ``pydantic``,
``numpy``, ``pandas``, ``scipy``, ``matplotlib``, ``seaborn``, and ``tqdm``.

To install the test dependencies as well:

.. code-block:: bash

   pip install .[test]

**Optional:** The ``pre_exp`` workflow can use ``molsym`` for symmetry number
detection. Install it separately if needed.

Basic Usage
-----------

All workflows are driven through a single CLI entry point with a JSON
configuration file:

.. code-block:: bash

   beep --config input.json

The JSON file must contain a ``"workflow"`` key that selects which workflow to
run. Example configuration files are provided in the ``examples/`` directory.

Discovering Workflows and Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List all available workflows with a short description of each:

.. code-block:: bash

   beep --workflows

Dump the full JSON schema for any workflow — this shows every field, its type,
default value, and a human-readable description:

.. code-block:: bash

   beep --schema sampling

This is the easiest way to create a new configuration file: start from the
schema output and fill in the required fields.

Workflows
---------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Workflow
     - Description
   * - ``sampling``
     - Generate binding-site candidates on water clusters (adaptive surface-anchored or spherical placement), RMSD-filter them and refine the geometries
   * - ``be_hess``
     - Submit binding energy (with BSSE) and Hessian computations for the optimized cluster structures
   * - ``extract``
     - Extract binding energies, apply ZPVE corrections, and generate summary tables and plots
   * - ``mbe``
     - Submit and monitor many-body-expansion binding energies on existing binding sites at a higher level of theory
   * - ``mbe_extract``
     - Assemble MBE binding energies and n-body decomposition tables, optionally borrowing a ZPVE correction from be_hess
   * - ``pre_exp``
     - Compute transition-state-theory pre-exponential desorption factors over a temperature range
   * - ``geom_benchmark``
     - Benchmark DFT geometry optimizations and per-step force RMSD along the reference trajectory against high-level reference data
   * - ``energy_benchmark``
     - Benchmark DFT binding energies against high-level reference values (CBS extrapolation), with optional gCP correction
   * - ``nm_sampling``
     - Benchmark DFT gradients on ± normal-mode displacements against a high-level (CCSD(T)) reference, with per-band force-RMSE ranking

Output Directory Structure
--------------------------

Each workflow creates an organized output directory under the molecule name:

.. code-block:: text

   N2/
     sampling/       # binding site structures, site_finder debug data, log
     be_hess/        # log, config copy
     extract/        # CSV tables, SVG plots, log
     pre_exp/        # pre-exponential factor results, log
     geom_benchmark/ # geometry benchmark results, log
     energy_benchmark/ # energy benchmark results, log

Every output folder contains a log file and a copy of the input configuration
for reproducibility.

Running Tests
-------------

.. code-block:: bash

   pip install .[test]
   pytest

The test suite covers core logic, models, and workflow integration without
requiring a running QCFractal server or psi4.

Periodic MLP workflows have moved to ``beep-mlp``. Existing JSON configurations
can be run with ``beep-mlp sampling_periodic --config input.json``,
``beep-mlp be_comp_periodic --config input.json`` or
``beep-mlp be_assemble_periodic --config input.json``.
