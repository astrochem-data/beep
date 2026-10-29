"""Tests for beep/workflows/mbe_extract.py (BE assembly core).

Ported from beep-mbe's test_assemble_be.py. Calls ``assemble_mbe_be`` directly
with a tmp output folder and monkeypatched ``qcf.get_collection``.
"""
import pandas as pd
import pytest
import qcelemental

constants = qcelemental.constants

from beep.adapters import qcfractal_adapter as qcf
from beep.workflows import mbe_extract as ex
from beep.core import mbe_be_tools as bt
from beep.models.mbe import MbeExtractConfig


class DummyRecord:
    def __init__(self, properties=None, return_result=None):
        self.properties = properties or {}
        self.return_result = return_result
        self.status = "COMPLETE"


class DummyManybodyDataset:
    def __init__(self, records, entry_names):
        self._records = records
        self.entry_names = entry_names

    def fetch_records(self, **kwargs):
        return None

    def get_record(self, entry_name, specification_name):
        return self._records.get((entry_name, specification_name))


class DummySinglepointDataset:
    def __init__(self, records):
        self._records = records

    def fetch_records(self, **kwargs):
        return None

    def get_record(self, entry_name, specification_name):
        return self._records.get((entry_name, specification_name))


def _properties_for(pref, e1=None, e2=None, e3=None, etot=None):
    results = {}
    if e1 is not None:
        results[f"{pref}_total_energy_through_1_body"] = e1
    if e2 is not None:
        results[f"{pref}_2_body_contribution_to_energy"] = e2
    if e3 is not None:
        results[f"{pref}_3_body_contribution_to_energy"] = e3
    if etot is not None:
        results[f"{pref}_total_energy"] = etot
    return {"results": results}


def _patch_collections(monkeypatch, mb_ds, sp_ds):
    def fake_get_collection(client, collection_type, name):
        if collection_type == "ManybodyDataset":
            return mb_ds
        if collection_type == "SinglepointDataset":
            return sp_ds
        raise KeyError(collection_type)
    monkeypatch.setattr(qcf, "get_collection", fake_get_collection)


def _make_config(specs, surface="surface"):
    return MbeExtractConfig(
        workflow="mbe_extract",
        opt_level_of_theory="hf3c_minix",
        small_molecule_collection="sp_ds",
        small_molecule="monomer",
        surface_model_collection="surface_ds",
        surface_model=surface,
        dataset="mbe_ds",
        entries=None,
        spec=specs,
        bsse=["vmfc"],
    )


def test_run_writes_report_and_csvs(tmp_path, monkeypatch):
    pref = "vmfc_corrected"
    spec = "spec"
    surface = "surface"
    entries = ["cluster-a", "cluster-b"]

    mb_records = {
        (surface, spec): DummyRecord(_properties_for(pref, e1=4.0, e2=0.5, e3=0.1, etot=4.6)),
        ("cluster-a", spec): DummyRecord(_properties_for(pref, e1=10.0, e2=1.5, e3=0.2, etot=11.7)),
        ("cluster-b", spec): DummyRecord(_properties_for(pref, e1=9.0, e2=None, e3=0.1, etot=10.2)),
    }
    sp_records = {("monomer", f"monomer_{spec}"): DummyRecord(return_result=1.2)}

    mb_ds = DummyManybodyDataset(mb_records, [surface, *entries])
    sp_ds = DummySinglepointDataset(sp_records)
    _patch_collections(monkeypatch, mb_ds, sp_ds)

    cfg = _make_config([spec], surface)
    out_path = ex.assemble_mbe_be(cfg, client=object(), res_folder=tmp_path)

    assert out_path.exists()
    data_dir = tmp_path / "be_data"
    total_csv = data_dir / "total_be.csv"
    assert total_csv.exists()
    assert (data_dir / "decomp__spec.csv").exists()
    assert (data_dir / "contrib__spec.csv").exists()

    df_total = pd.read_csv(total_csv, index_col=0)
    assert list(df_total.columns) == [spec]
    assert list(df_total.index) == entries

    total_expected = bt.compute_be_values(
        {"e1": 10.0, "e2": 1.5, "e3": 0.2, "etot": 11.7},
        {"e1": 4.0, "e2": 0.5, "e3": 0.1, "etot": 4.6},
        1.2,
    )["be_total"]
    assert df_total.loc["cluster-a", spec] == pytest.approx(total_expected * constants.hartree2kcalmol)

    df_decomp = pd.read_csv(data_dir / "decomp__spec.csv", index_col=0)
    assert list(df_decomp.columns) == ["BE_1b", "BE_2b", "BE_3b"]
    be_1b = df_decomp.loc["cluster-a", "BE_1b"]
    be_2b = df_decomp.loc["cluster-a", "BE_2b"]
    be_3b = df_decomp.loc["cluster-a", "BE_3b"]
    assert be_2b == pytest.approx(be_1b + (-(1.5 - 0.5) * constants.hartree2kcalmol))
    assert be_3b == pytest.approx(be_2b + (-(0.2 - 0.1) * constants.hartree2kcalmol))
    assert pd.isna(df_decomp.loc["cluster-b", "BE_2b"])

    report_text = out_path.read_text()
    assert "BSSE scheme: vmfc" in report_text
    assert "Per-body contributions" in report_text
    assert "\t" not in report_text
    # No ZPVE table when zpve config is absent.
    assert not (data_dir / "total_be_zpve.csv").exists()
    assert "ZPVE" not in report_text

    # Convergence / truncation-error output (always emitted).
    conv_csv = data_dir / "convergence__spec.csv"
    assert conv_csv.exists()
    df_conv = pd.read_csv(conv_csv, index_col=0)
    assert list(df_conv.columns) == [
        "BE_total", "n_body_max", "delta_last", "ratio_r",
        "error_bar", "rel_error", "converged",
    ]
    # cluster-a is a full 3-body site -> a real geometric bar.
    assert df_conv.loc["cluster-a", "n_body_max"] == 3
    assert df_conv.loc["cluster-a", "error_bar"] > 0
    # cluster-b lacks the 2-body term -> not estimable (n/a).
    assert pd.isna(df_conv.loc["cluster-b", "error_bar"])
    assert "MBE truncation error" in report_text


def test_zpve_enabled_writes_corrected_table(tmp_path, monkeypatch):
    pref = "vmfc_corrected"
    spec = "spec"
    surface = "surface"
    entries = ["cluster-a"]

    mb_records = {
        (surface, spec): DummyRecord(_properties_for(pref, e1=4.0, e2=0.5, e3=0.1, etot=4.6)),
        ("cluster-a", spec): DummyRecord(_properties_for(pref, e1=10.0, e2=1.5, e3=0.2, etot=11.7)),
    }
    sp_records = {("monomer", f"monomer_{spec}"): DummyRecord(return_result=1.2)}
    mb_ds = DummyManybodyDataset(mb_records, [surface, *entries])
    sp_ds = DummySinglepointDataset(sp_records)
    _patch_collections(monkeypatch, mb_ds, sp_ds)

    # Stub the read-only ZPVE borrow to a known, non-symmetric per-site value
    # so the sign of the correction is actually checked below.
    delta_zpve = 1.7
    monkeypatch.setattr(
        bt, "borrow_zpve_corrections",
        lambda *a, **k: pd.Series({"cluster-a": delta_zpve}, name="Delta_ZPVE"),
    )

    cfg = MbeExtractConfig(
        workflow="mbe_extract",
        opt_level_of_theory="hf3c_minix",
        small_molecule_collection="sp_ds",
        small_molecule="monomer",
        surface_model_collection="surface_ds",
        surface_model=surface,
        dataset="mbe_ds",
        spec=[spec],
        bsse=["vmfc"],
        zpve={"enabled": True, "hessian_clusters": ["cd5"]},
    )
    out_path = ex.assemble_mbe_be(cfg, client=object(), res_folder=tmp_path)

    data_dir = tmp_path / "be_data"
    zpve_csv = data_dir / "total_be_zpve.csv"
    assert zpve_csv.exists()
    df_zpve = pd.read_csv(zpve_csv, index_col=0)
    assert f"{spec}+ZPVE" in df_zpve.columns
    assert "Delta_ZPVE" in df_zpve.columns
    df_total = pd.read_csv(data_dir / "total_be.csv", index_col=0)
    # MBE BEs are positive-convention and Delta_ZPVE = ZPVE(complex) - ZPVE(parts)
    # is positive, so the ZPVE-corrected BE must be BE - Delta_ZPVE (smaller).
    be_uncorr = df_total.loc["cluster-a", spec]
    be_corr = df_zpve.loc["cluster-a", f"{spec}+ZPVE"]
    assert be_corr == pytest.approx(be_uncorr - delta_zpve)
    assert be_corr < be_uncorr
    assert be_corr != pytest.approx(be_uncorr + delta_zpve)
    assert df_zpve.loc["cluster-a", "Delta_ZPVE"] == pytest.approx(delta_zpve)
    assert "ZPVE correction" in out_path.read_text()


def test_run_supports_multiple_specs(tmp_path, monkeypatch):
    pref = "vmfc_corrected"
    specs = ["spec-a", "spec-b"]
    surface = "surface"
    entries = ["cluster-a"]

    mb_records = {}
    sp_records = {}
    for spec in specs:
        mb_records[(surface, spec)] = DummyRecord(_properties_for(pref, e1=4.0, e2=0.5, e3=0.1, etot=4.6))
        mb_records[(entries[0], spec)] = DummyRecord(_properties_for(pref, e1=10.0, e2=1.5, e3=0.2, etot=11.7))
        sp_records[("monomer", f"monomer_{spec}")] = DummyRecord(return_result=1.2)

    mb_ds = DummyManybodyDataset(mb_records, [surface, *entries])
    sp_ds = DummySinglepointDataset(sp_records)
    _patch_collections(monkeypatch, mb_ds, sp_ds)

    cfg = _make_config(specs, surface)
    out_path = ex.assemble_mbe_be(cfg, client=object(), res_folder=tmp_path)

    data_dir = tmp_path / "be_data"
    df_total = pd.read_csv(data_dir / "total_be.csv", index_col=0)
    assert list(df_total.columns) == specs
    for spec in specs:
        assert (data_dir / f"decomp__{spec}.csv").exists()
        assert (data_dir / f"contrib__{spec}.csv").exists()


@pytest.mark.parametrize("scheme,pref", bt.BSSE_PREFIX.items())
@pytest.mark.parametrize("order", [4, 5, 10])
def test_higher_order_csvs_and_convergence(tmp_path, monkeypatch, scheme, pref, order):
    """A small third-order term must not hide a growing final increment."""
    def props(base, increments):
        return {"results": {
            f"{pref}_total_energy_through_1_body": base,
            **{f"{pref}_{n}_body_contribution_to_energy": v for n, v in increments.items()},
            f"{pref}_total_energy": base + sum(increments.values()),
        }}

    surface = {n: 0.01 for n in range(2, order + 1)}
    # BE increments shrink until the final order, then grow by a factor of 2.
    increments = {n: 0.1 ** (n - 2) for n in range(2, order)}
    increments[order] = 2 * increments[order - 1]
    complex_terms = {n: surface[n] - increments[n] for n in surface}
    records = {
        ("surface", "spec"): DummyRecord(props(-10., surface)),
        ("site", "spec"): DummyRecord(props(-12., complex_terms)),
    }
    _patch_collections(monkeypatch, DummyManybodyDataset(records, ["surface", "site"]),
                       DummySinglepointDataset({("monomer", "monomer_spec"): DummyRecord(return_result=-2.)}))
    cfg = _make_config(["spec"]).model_copy(update={"bsse": [scheme]})
    report = ex.assemble_mbe_be(cfg, object(), tmp_path)
    data = tmp_path / "be_data"
    cumulative = pd.read_csv(data / "decomp__spec.csv", index_col=0)
    contrib = pd.read_csv(data / "contrib__spec.csv", index_col=0)
    total = pd.read_csv(data / "total_be.csv", index_col=0)
    conv = pd.read_csv(data / "convergence__spec.csv", index_col=0)
    assert list(cumulative.columns) == [f"BE_{n}b" for n in range(1, order + 1)]
    assert list(contrib.columns) == [f"BE_{n}b_contrib" for n in range(1, order + 1)]
    for n in range(2, order + 1):
        assert contrib.loc["site", f"BE_{n}b_contrib"] == pytest.approx(
            increments[n] * constants.hartree2kcalmol, abs=1e-8)
    assert cumulative.loc["site", f"BE_{order}b"] == pytest.approx(total.loc["site", "spec"])
    assert conv.loc["site", "n_body_max"] == order
    assert conv.loc["site", "ratio_r"] == pytest.approx(2.)
    assert not conv.loc["site", "converged"]
    assert pd.isna(conv.loc["site", "error_bar"])
    assert f"BE_{order}b_contrib" in report.read_text()


def test_missing_higher_reference_is_not_zero_or_lower_order_convergence():
    sup = {"e1": -12., "e2": -.2, "e3": -.01, "e4": -.002, "etot": -12.212}
    surf = {"e1": -10., "e2": -.1, "e3": -.005, "etot": -10.105}
    values = bt.compute_be_values(sup, surf, -2.)
    assert values["be_4"] is None
    assert pd.isna(bt.build_cumulative(values)["BE_4b"])
    assert pd.isna(bt.build_contributions(values)["BE_4b_contrib"])
    conv = bt.compute_convergence(values)
    assert conv["n_body_max"] == 4
    assert conv["error_bar"] is None and conv["converged"] is None


def test_extract_reports_missing_increment_and_ignores_other_schemes():
    rec = DummyRecord({"results": {
        "nocp_total_energy_through_4_body": -12.,
        "nocp_total_energy": -12.,
        "nocp_3_body_contribution_to_energy": -.1,
        "nocp_4_body_contribution_to_gradient": [0., 0., 0.],
        "cp_corrected_5_body_contribution_to_energy": -.02,
    }})
    result = bt.extract_mbe_components(rec, "nocp")
    assert result["e4"] is None and result["e3"] == -.1
    assert "e5" not in result
    assert result["etot"] == -12.


def test_cumulative_missing_intermediate_order_propagates():
    values = {"be_le1": 1., "be_2": 2., "be_3": None, "be_4": .1, "be_5": .01}
    result = bt.build_cumulative(values)
    assert result["BE_2b"] == 3.
    assert all(pd.isna(result[f"BE_{n}b"]) for n in [3, 4, 5])
    assert bt.build_contributions(values)["BE_5b_contrib"] == .01
    assert bt.compute_convergence(values)["error_bar"] is None


def test_mixed_orders_and_missing_terms_in_workflow(tmp_path, monkeypatch):
    pref = "vmfc_corrected"
    records, sp_records = {}, {}
    for spec in ["three", "four"]:
        surface = _properties_for(pref, e1=-10., e2=-.1, e3=-.01, etot=-10.11)
        complete = _properties_for(pref, e1=-12., e2=-.2, e3=-.02, etot=-12.22)
        missing = _properties_for(pref, e1=-12., e2=-.2, e3=-.02, etot=-12.22)
        if spec == "four":
            surface["results"][f"{pref}_4_body_contribution_to_energy"] = -.001
            complete["results"][f"{pref}_4_body_contribution_to_energy"] = -.002
            surface["results"][f"{pref}_total_energy"] -= .001
            complete["results"][f"{pref}_total_energy"] -= .002
        records[("surface", spec)] = DummyRecord(surface)
        records[("complete", spec)] = DummyRecord(complete)
        records[("missing", spec)] = DummyRecord(missing)
        sp_records[("monomer", f"monomer_{spec}")] = DummyRecord(return_result=-2.)
    _patch_collections(monkeypatch, DummyManybodyDataset(records, ["surface", "missing", "complete"]),
                       DummySinglepointDataset(sp_records))
    report = ex.assemble_mbe_be(_make_config(["three", "four"]), object(), tmp_path)
    data = tmp_path / "be_data"
    three = pd.read_csv(data / "decomp__three.csv", index_col=0)
    four = pd.read_csv(data / "decomp__four.csv", index_col=0)
    conv = pd.read_csv(data / "convergence__four.csv", index_col=0)
    assert list(three.columns) == ["BE_1b", "BE_2b", "BE_3b"]
    assert list(four.columns) == ["BE_1b", "BE_2b", "BE_3b", "BE_4b"]
    assert pd.isna(four.loc["missing", "BE_4b"])
    assert conv.loc["missing", "n_body_max"] == 4
    assert pd.isna(conv.loc["missing", "converged"])
    assert conv.loc["complete", "ratio_r"] == pytest.approx(.1)
    assert conv.loc["complete", "converged"]
    assert "n/a (not estimable)" in report.read_text()
