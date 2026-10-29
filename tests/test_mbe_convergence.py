"""Tests for the MBE truncation-error estimate (compute_convergence).

Symmetric geometric-tail bar; no signed BE_inf. Values checked against the
worked numbers from real qcf-astrochem-test data.
"""
import math

import pytest

from beep.core.mbe_be_tools import compute_convergence, format_convergence_table


# Real kcal/mol increments (be_le1 deformation, be_2, be_3, be_total).
H2O = {"be_le1": -0.62011498, "be_2": 4.64679471, "be_3": 0.31832859, "be_total": 4.34500832}
CO = {"be_le1": -0.07406188, "be_2": 2.13521454, "be_3": -0.07778028, "be_total": 1.98337238}
TWO_BODY = {"be_le1": -0.05, "be_2": 1.33624634, "be_3": None, "be_total": 1.33624634}
NON_CONV = {"be_le1": 0.0, "be_2": 1.0, "be_3": 1.2, "be_total": 2.2}


def test_three_body_geometric_bar():
    c = compute_convergence(H2O, tol=0.05)
    assert c["n_body_max"] == 3
    assert c["ratio_r"] == pytest.approx(0.31832859 / 4.64679471)
    # bar = |d3| * r/(1-r)
    r = 0.31832859 / 4.64679471
    assert c["error_bar"] == pytest.approx(abs(0.31832859) * r / (1 - r))
    assert c["error_bar"] == pytest.approx(0.02341, abs=1e-4)
    assert c["converged"] is True


def test_sign_flip_uses_magnitude_ratio():
    # 3-body increment is negative; ratio/bar are magnitudes, still positive.
    c = compute_convergence(CO, tol=0.05)
    assert c["delta_last"] == pytest.approx(-0.07778028)
    assert c["ratio_r"] > 0
    assert c["error_bar"] == pytest.approx(0.00294, abs=1e-4)
    assert c["converged"] is True


def test_two_body_run_is_not_estimable():
    c = compute_convergence(TWO_BODY, tol=0.05)
    assert c["n_body_max"] == 2
    assert c["error_bar"] is None
    assert c["ratio_r"] is None
    assert c["converged"] is None


def test_non_shrinking_series_flagged_not_converged():
    c = compute_convergence(NON_CONV, tol=0.05)
    assert c["ratio_r"] == pytest.approx(1.2)
    assert c["error_bar"] is None
    assert c["converged"] is False


def test_tol_controls_flag():
    # H2O rel_error ~0.0054; a very tight tol should flip converged to False.
    assert compute_convergence(H2O, tol=0.001)["converged"] is False
    assert compute_convergence(H2O, tol=0.05)["converged"] is True


def test_no_signed_be_inf_key():
    # The result must not expose a signed corrected BE (design decision).
    c = compute_convergence(H2O, tol=0.05)
    assert "be_inf" not in c
    assert "BE_inf" not in c


def test_format_convergence_table_handles_na():
    rows = {
        "site_3b": {"BE_total": 4.345, "error_bar": 0.0234, "n_body_max": 3, "converged": True},
        "site_2b": {"BE_total": 1.336, "error_bar": None, "n_body_max": 2, "converged": None},
    }
    txt = format_convergence_table(rows)
    assert "± 0.023400" in txt
    assert "n/a (2b)" in txt
    assert "yes" in txt


@pytest.mark.parametrize("last", [.02, -.02, 0.])
def test_four_body_tail_uses_fourth_and_third_orders(last):
    values = {"be_le1": -.1, "be_2": 1., "be_3": .1, "be_4": last, "be_total": 1. + last}
    c = compute_convergence(values)
    r = abs(last / .1)
    assert c["n_body_max"] == 4
    assert c["delta_last"] == last
    assert c["ratio_r"] == pytest.approx(r)
    assert c["error_bar"] == pytest.approx(abs(last) * r / (1 - r))
    assert c["converged"] is True


@pytest.mark.parametrize("previous", [0., None, float("nan"), float("inf")])
def test_four_body_tail_requires_finite_nonzero_previous(previous):
    c = compute_convergence({"be_2": 1., "be_3": previous, "be_4": .02, "be_total": 1.})
    assert c["n_body_max"] == 4
    assert c["error_bar"] is None and c["converged"] is None
