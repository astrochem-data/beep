API Reference
=============

Core (``beep.core``)
--------------------

Pure computational logic with no external server dependencies.

.. automodule:: beep.core.molecule_sampler
   :members:
   :undoc-members:

.. automodule:: beep.core.sampling
   :members:
   :undoc-members:

.. automodule:: beep.core.stoichiometry
   :members:
   :undoc-members:

.. automodule:: beep.core.zpve
   :members:
   :undoc-members:

.. automodule:: beep.core.cbs_extrapolation
   :members:
   :undoc-members:

.. automodule:: beep.core.pre_exponential
   :members:
   :undoc-members:

.. automodule:: beep.core.dft_functionals
   :members:
   :undoc-members:

.. automodule:: beep.core.be_tools
   :members:
   :undoc-members:

.. automodule:: beep.core.benchmark_utils
   :members:
   :undoc-members:

.. automodule:: beep.core.plotting_utils
   :members:
   :undoc-members:

.. automodule:: beep.core.logging_utils
   :members:
   :undoc-members:

.. automodule:: beep.core.errors
   :members:
   :undoc-members:

.. automodule:: beep.core.exceptions
   :members:
   :undoc-members:

.. automodule:: beep.core.mbe_fragmentation
   :members:
   :undoc-members:

.. automodule:: beep.core.mbe_be_tools
   :members:
   :undoc-members:

.. automodule:: beep.core.normal_mode_sampling
   :members:
   :undoc-members:

.. automodule:: beep.core.nm_sampling_workflow
   :members:
   :undoc-members:

.. automodule:: beep.core.trajectory_workflow
   :members:
   :undoc-members:

.. automodule:: beep.core.trajectory_metrics
   :members:
   :undoc-members:

Models (``beep.models``)
------------------------

Pydantic configuration models for each workflow.

.. automodule:: beep.models.base
   :members:
   :undoc-members:

.. automodule:: beep.models.sampling
   :members:
   :undoc-members:

.. automodule:: beep.models.be_hess
   :members:
   :undoc-members:

.. automodule:: beep.models.extract
   :members:
   :undoc-members:

.. automodule:: beep.models.pre_exp
   :members:
   :undoc-members:

.. automodule:: beep.models.geom_benchmark
   :members:
   :undoc-members:

.. automodule:: beep.models.energy_benchmark
   :members:
   :undoc-members:

.. automodule:: beep.models.mbe
   :members:
   :undoc-members:

.. automodule:: beep.models.nm_sampling
   :members:
   :undoc-members:

Adapters (``beep.adapters``)
----------------------------

Thin wrappers around external services (QCFractal server I/O).

.. automodule:: beep.adapters.qcfractal_adapter
   :members:
   :undoc-members:

Workflows (``beep.workflows``)
------------------------------

Orchestration layer that ties core logic and adapters together.

.. automodule:: beep.workflows.sampling
   :members:
   :undoc-members:

.. automodule:: beep.workflows.be_hess
   :members:
   :undoc-members:

.. automodule:: beep.workflows.extract
   :members:
   :undoc-members:

.. automodule:: beep.workflows.pre_exp
   :members:
   :undoc-members:

.. automodule:: beep.workflows.geom_benchmark
   :members:
   :undoc-members:

.. automodule:: beep.workflows.energy_benchmark
   :members:
   :undoc-members:

.. automodule:: beep.workflows.mbe
   :members:
   :undoc-members:

.. automodule:: beep.workflows.mbe_extract
   :members:
   :undoc-members:

.. automodule:: beep.workflows.nm_sampling
   :members:
   :undoc-members:

CLI (``beep.cli``)
------------------

Command-line interface entry point.

.. automodule:: beep.cli
   :members:
   :undoc-members:
