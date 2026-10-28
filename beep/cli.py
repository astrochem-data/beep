"""
BEEP CLI — single JSON-config entry point for all workflows.

Usage:
    beep --config input.json
    beep --workflows
    beep --schema sampling

The JSON file must contain a "workflow" key that selects which workflow to run.
Valid workflow values: sampling, be_hess, extract,
mbe, mbe_extract, pre_exp,
geom_benchmark, energy_benchmark, nm_sampling
"""
import argparse
import json
import sys
import logging
from pathlib import Path

from pydantic import ValidationError

from .models import (
    SamplingConfig,
    BeHessConfig,
    ExtractConfig,
    PreExpConfig,
    GeomBenchmarkConfig,
    EnergyBenchmarkConfig,
    NmSamplingConfig,
    MbeConfig,
    MbeExtractConfig,
)
from .adapters.qcfractal_adapter import connect

WORKFLOW_MODELS = {
    "sampling": SamplingConfig,
    "be_hess": BeHessConfig,
    "extract": ExtractConfig,
    "pre_exp": PreExpConfig,
    "geom_benchmark": GeomBenchmarkConfig,
    "energy_benchmark": EnergyBenchmarkConfig,
    "nm_sampling": NmSamplingConfig,
    "mbe": MbeConfig,
    "mbe_extract": MbeExtractConfig,
}


def _print_workflows():
    """Print a table of available workflows with descriptions."""
    print("Available workflows:\n")
    print(f"  {'Name':<20} Description")
    print(f"  {'----':<20} -----------")
    for name, model in WORKFLOW_MODELS.items():
        # first paragraph line only; the full docstring is in `beep --schema`
        doc = (model.__doc__ or "").strip().splitlines()[0].strip().rstrip(".") if (model.__doc__ or "").strip() else ""
        print(f"  {name:<20} {doc}")
    print()
    print("Use 'beep --schema <workflow>' to see the full JSON schema for a workflow.")


def _print_schema(workflow_name):
    """Print the JSON schema for a workflow model."""
    if workflow_name not in WORKFLOW_MODELS:
        print(
            f"Error: unknown workflow '{workflow_name}'. "
            f"Valid options: {', '.join(WORKFLOW_MODELS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    model = WORKFLOW_MODELS[workflow_name]
    print(json.dumps(model.model_json_schema(), indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="beep",
        description="BEEP — Binding Energy Evaluation Platform",
        epilog=(
            "examples:\n"
            "  beep --config input.json      Run a workflow from a JSON config\n"
            "  beep --workflows              List available workflows\n"
            "  beep --schema sampling         Show JSON schema for a workflow\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--workflows",
        action="store_true",
        help="List available workflows and exit",
    )
    parser.add_argument(
        "--schema",
        metavar="WORKFLOW",
        help="Print the JSON schema for WORKFLOW and exit",
    )
    args = parser.parse_args()

    # --workflows: print list and exit
    if args.workflows:
        _print_workflows()
        sys.exit(0)

    # --schema: print schema and exit
    if args.schema is not None:
        _print_schema(args.schema)
        sys.exit(0)

    # Otherwise --config is required
    if args.config is None:
        parser.error("--config is required (or use --workflows / --schema)")

    # Load and parse config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(config_path.read_text())
    workflow = raw.get("workflow")

    if workflow not in WORKFLOW_MODELS:
        print(
            f"Error: unknown workflow '{workflow}'. "
            f"Valid options: {', '.join(WORKFLOW_MODELS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate with the appropriate Pydantic model
    try:
        config = WORKFLOW_MODELS[workflow](**raw)
    except ValidationError as e:
        print(
            f"Error: invalid configuration for workflow '{workflow}' "
            f"in {config_path} ({e.error_count()} error(s)):",
            file=sys.stderr,
        )
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
            print(f"  {loc}: {err.get('msg')}", file=sys.stderr)
        print(
            f"Use 'beep --schema {workflow}' to see the expected fields.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set up console-only logging (workflows add file handlers per output folder)
    logger = logging.getLogger("beep")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    logger.info(f"Running workflow: {workflow}")

    # Connect to QCFractal server
    client = connect(
        address=config.server.address,
        username=config.server.username,
        password=config.server.password,
        verify=config.server.verify,
    )

    # Import and dispatch to the appropriate workflow
    if workflow == "sampling":
        from .workflows.sampling import run
    elif workflow == "be_hess":
        from .workflows.be_hess import run
    elif workflow == "extract":
        from .workflows.extract import run
    elif workflow == "pre_exp":
        from .workflows.pre_exp import run
    elif workflow == "geom_benchmark":
        from .workflows.geom_benchmark import run
    elif workflow == "energy_benchmark":
        from .workflows.energy_benchmark import run
    elif workflow == "nm_sampling":
        from .workflows.nm_sampling import run
    elif workflow == "mbe":
        from .workflows.mbe import run
    elif workflow == "mbe_extract":
        from .workflows.mbe_extract import run

    run(config, client)

    logger.info(f"\nWorkflow '{workflow}' finished successfully.")


if __name__ == "__main__":
    main()
