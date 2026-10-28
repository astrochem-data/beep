"""Tests for beep/cli.py."""
import json
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

from beep.cli import main, WORKFLOW_MODELS


def test_main_missing_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["beep"])
    with pytest.raises(SystemExit):
        main()


def test_main_nonexistent_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_invalid_workflow(monkeypatch, tmp_path):
    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text(json.dumps({"workflow": "bogus"}))
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(cfg_file)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("beep.cli.connect")
def test_main_valid_dispatch(mock_connect, monkeypatch, tmp_path):
    mock_client = MagicMock()
    mock_connect.return_value = mock_client

    cfg_file = tmp_path / "sampling.json"
    cfg_file.write_text(json.dumps({
        "workflow": "sampling",
        "molecule": "CO",
        "sampling_level_of_theory": {"method": "gfn2-xtb", "program": "xtb"},
        "refinement_level_of_theory": {"method": "bhandhlyp", "basis": "def2-svp", "program": "psi4"},
    }))
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(cfg_file)])

    # Create a fake module with a run function to intercept the lazy import
    mock_run = MagicMock()
    fake_module = types.ModuleType("beep.workflows.sampling")
    fake_module.run = mock_run

    with patch.dict(sys.modules, {"beep.workflows.sampling": fake_module}):
        main()

    mock_connect.assert_called_once()
    mock_run.assert_called_once()


def test_workflow_models_keys():
    expected = {
        "sampling",
        "be_hess", "extract", "pre_exp",
        "geom_benchmark", "energy_benchmark", "nm_sampling",
        "mbe", "mbe_extract",
    }
    assert set(WORKFLOW_MODELS.keys()) == expected


def test_pre_existing_workflow_models_unchanged():
    """The seven original workflows must remain registered with their models."""
    from beep.models import (
        SamplingConfig, BeHessConfig, ExtractConfig, PreExpConfig,
        GeomBenchmarkConfig, EnergyBenchmarkConfig, NmSamplingConfig,
    )
    assert WORKFLOW_MODELS["sampling"] is SamplingConfig
    assert WORKFLOW_MODELS["be_hess"] is BeHessConfig
    assert WORKFLOW_MODELS["extract"] is ExtractConfig
    assert WORKFLOW_MODELS["pre_exp"] is PreExpConfig
    assert WORKFLOW_MODELS["geom_benchmark"] is GeomBenchmarkConfig
    assert WORKFLOW_MODELS["energy_benchmark"] is EnergyBenchmarkConfig
    assert WORKFLOW_MODELS["nm_sampling"] is NmSamplingConfig


@patch("beep.cli.connect")
def test_main_dispatch_mbe(mock_connect, monkeypatch, tmp_path):
    mock_connect.return_value = MagicMock()

    cfg_file = tmp_path / "mbe.json"
    cfg_file.write_text(json.dumps({
        "workflow": "mbe",
        "opt_level_of_theory": "hf3c_minix",
        "opt_dataset": "H2CO_cd5",
        "small_molecule_collection": "astro_molecules",
        "small_molecule": "H2CO",
        "surface_model_collection": "co2_x",
        "surface_model": "cd5_01",
        "env_unit_len": 3,
        "dataset": "H2CO_cd5",
        "spec": ["scf_adz_vmfc"],
        "bsse": ["vmfc"],
        "levels": [{"index": 1, "method": "scf", "basis": "aug-cc-pvdz"}],
    }))
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(cfg_file)])

    mock_run = MagicMock()
    fake_module = types.ModuleType("beep.workflows.mbe")
    fake_module.run = mock_run

    with patch.dict(sys.modules, {"beep.workflows.mbe": fake_module}):
        main()

    mock_run.assert_called_once()


@patch("beep.cli.connect")
def test_main_dispatch_mbe_extract(mock_connect, monkeypatch, tmp_path):
    mock_connect.return_value = MagicMock()

    cfg_file = tmp_path / "mbe_extract.json"
    cfg_file.write_text(json.dumps({
        "workflow": "mbe_extract",
        "opt_level_of_theory": "hf3c_minix",
        "small_molecule_collection": "astro_molecules",
        "small_molecule": "H2CO",
        "surface_model_collection": "co2_x",
        "surface_model": "cd5_01",
        "dataset": "H2CO_cd5",
        "spec": ["scf_adz_vmfc"],
        "bsse": ["vmfc"],
    }))
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(cfg_file)])

    mock_run = MagicMock()
    fake_module = types.ModuleType("beep.workflows.mbe_extract")
    fake_module.run = mock_run

    with patch.dict(sys.modules, {"beep.workflows.mbe_extract": fake_module}):
        main()

    mock_run.assert_called_once()


def test_schema_prints_valid_json(monkeypatch, capsys):
    """--schema must use the pydantic v2 API and emit parseable JSON."""
    monkeypatch.setattr(sys, "argv", ["beep", "--schema", "geom_benchmark"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "GeomBenchmarkConfig"
    assert "qc_keywords" in schema["properties"]
    # deprecated null-only alias is loadable but not advertised
    assert "dft_optimization_keyword" not in schema["properties"]


def test_schema_does_not_use_deprecated_schema_json(monkeypatch, capsys):
    import warnings
    monkeypatch.setattr(sys, "argv", ["beep", "--schema", "sampling"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(SystemExit):
            main()
    json.loads(capsys.readouterr().out)


@patch("beep.cli.connect")
def test_main_invalid_config_prints_clean_error(mock_connect, monkeypatch, tmp_path, capsys):
    """A config that fails model validation must exit 1 with a readable
    per-field message instead of a pydantic traceback."""
    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text(json.dumps({
        "workflow": "geom_benchmark",
        "opt_dataset": "ds",
        # benchmark_structures missing; deprecated name with a keyword ID
        "dft_optimization_keyword": 7,
    }))
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(cfg_file)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "invalid configuration for workflow 'geom_benchmark'" in err
    assert "benchmark_structures" in err
    assert "dft_optimization_keyword" in err
    assert "qc_keywords" in err
    assert "Traceback" not in err
    mock_connect.assert_not_called()
