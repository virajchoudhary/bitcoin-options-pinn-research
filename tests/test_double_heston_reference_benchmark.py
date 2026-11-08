"""Tests for the independent adaptive-quadrature benchmark seam."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.constants import CALL_OPTION, PARAMETER_NAMES
from src.constraints import validate_parameters
from src.double_heston import price_double_heston_option
from src.double_heston_reference import (
    REFERENCE_EPSABS,
    REFERENCE_EPSREL,
    REFERENCE_LIMIT,
    reference_double_heston_characteristic_function,
    reference_double_heston_option,
)

from tests.synthetic_fixtures import BENCHMARK


@pytest.fixture(scope="module")
def fixture() -> dict:
    return deepcopy(BENCHMARK)


def test_reference_cf_at_zero_is_one(fixture: dict) -> None:
    case = fixture["cases"][0]
    actual = reference_double_heston_characteristic_function(
        0.0, case["spot"], case["maturity"], case["rate"], case["dividend_yield"], case["parameters"]
    )
    assert actual == pytest.approx(1.0 + 0.0j, abs=1e-13)


def test_controlled_reference_price_has_full_reliable_diagnostics(fixture: dict) -> None:
    case = fixture["cases"][0]
    price, diagnostic = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    assert np.isfinite(price)
    assert diagnostic["reliable"] is True
    assert diagnostic["p1"]["absolute_error_estimate"] <= diagnostic["p1"]["tolerance"]
    assert diagnostic["p2"]["absolute_error_estimate"] <= diagnostic["p2"]["tolerance"]
    assert "evaluation_count" in diagnostic["p1"] and "subdivisions" in diagnostic["p2"]
    assert diagnostic["epsabs"] == REFERENCE_EPSABS
    assert diagnostic["epsrel"] == REFERENCE_EPSREL
    assert diagnostic["limit"] == REFERENCE_LIMIT


def test_reference_no_arbitrage_and_parity(fixture: dict) -> None:
    call_case = next(case for case in fixture["cases"] if case["case_id"] == "B_2_call")
    call, _ = reference_double_heston_option(
        call_case["spot"], call_case["strike"], call_case["maturity"], call_case["rate"], call_case["dividend_yield"],
        "call", call_case["parameters"],
    )
    put, _ = reference_double_heston_option(
        call_case["spot"], call_case["strike"], call_case["maturity"], call_case["rate"], call_case["dividend_yield"],
        "put", call_case["parameters"],
    )
    discounted_spot = call_case["spot"] * np.exp(-call_case["dividend_yield"] * call_case["maturity"])
    discounted_strike = call_case["strike"] * np.exp(-call_case["rate"] * call_case["maturity"])
    assert max(discounted_spot - discounted_strike, 0.0) <= call <= discounted_spot
    assert max(discounted_strike - discounted_spot, 0.0) <= put <= discounted_strike
    assert call - put == pytest.approx(discounted_spot - discounted_strike, abs=1e-10)


@pytest.mark.parametrize("case_id", ["A_1_call", "D_2_put"])
def test_production_reference_closeness_uses_frozen_combined_tolerance(fixture: dict, case_id: str) -> None:
    case = next(item for item in fixture["cases"] if item["case_id"] == case_id)
    reference, diagnostic = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    assert diagnostic["reliable"]
    tolerance = fixture["immutable_tolerances"]["absolute"] + fixture["immutable_tolerances"]["relative"] * abs(reference)
    for nodes in (64, 96):
        production = price_double_heston_option(
            case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
            case["option_type"], case["parameters"], node_count=nodes,
        )
        assert abs(production - reference) <= tolerance


def test_fixture_is_valid_and_has_declared_coverage(fixture: dict) -> None:
    assert fixture["canonical_parameter_order"] == PARAMETER_NAMES
    assert fixture["immutable_tolerances"] == {
        "absolute": 2e-5, "relative": 2e-6, "no_arbitrage": 1e-8,
        "parity": 1e-10, "reference_epsabs": REFERENCE_EPSABS,
        "reference_epsrel": REFERENCE_EPSREL, "reference_limit": REFERENCE_LIMIT,
    }
    assert len(fixture["cases"]) == 36
    required_fields = {
        "case_id", "spot", "strike", "maturity", "rate", "dividend_yield",
        "option_type", "parameters", "case_category", "generation_metadata",
    }
    assert all(required_fields <= set(case) for case in fixture["cases"])
    assert all(set(case["generation_metadata"]) >= {"pair_id", "creation_method", "design_dimensions"} for case in fixture["cases"])
    assert all(len(case["parameters"]) == 10 and np.isfinite(case["parameters"]).all() for case in fixture["cases"])
    assert {case["option_type"] for case in fixture["cases"]} == {"call", "put"}
    assert {"short", "medium", "long"} == {
        "short" if case["maturity"] <= .25 else "medium" if case["maturity"] <= 1 else "long"
        for case in fixture["cases"]
    }
    assert all(validate_parameters(case["parameters"])["is_valid"] for case in fixture["cases"])
    categories = {case["case_category"] for case in fixture["cases"]}
    assert any("low_total_variance" in category for category in categories)
    assert any("high_total_variance" in category for category in categories)
    assert any("weak_negative_skew" in category for category in categories)
    assert any("strong_negative_skew" in category for category in categories)
    assert "moderately_close_feller_boundary" in categories
    assert "safely_interior_low_total_variance" in categories
    assert len({case["spot"] for case in fixture["cases"]}) > 1
    assert len({case["rate"] for case in fixture["cases"]}) > 1
    assert len({case["dividend_yield"] for case in fixture["cases"]}) > 1
    for option_type in ("call", "put"):
        labels = {_moneyness_bucket(case)[0] for case in fixture["cases"] if case["option_type"] == option_type}
        assert labels == {"ITM", "ATM", "OTM"}
    pairs: dict[str, list[dict]] = {}
    for case in fixture["cases"]:
        pairs.setdefault(case["generation_metadata"]["pair_id"], []).append(case)
    assert len(pairs) == 18
    for cases in pairs.values():
        assert len(cases) == 2
        assert {case["option_type"] for case in cases} == {"call", "put"}
        assert cases[0]["parameters"] == cases[1]["parameters"]
        for name in ("spot", "strike", "maturity", "rate", "dividend_yield"):
            assert cases[0][name] == cases[1][name]


def test_fixture_cases_have_no_silent_reference_failure(fixture: dict) -> None:
    for case in fixture["cases"]:
        _, diagnostic = reference_double_heston_option(
            case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
            case["option_type"], case["parameters"],
        )
        assert diagnostic["reliable"] and diagnostic["failure"] is None
















def test_reference_and_production_are_statically_independent() -> None:
    root = Path(__file__).resolve().parents[1]
    reference_source = (root / "src" / "double_heston_reference.py").read_text(encoding="utf-8")
    production_source = (root / "src" / "double_heston.py").read_text(encoding="utf-8")
    assert "from .double_heston import" not in reference_source
    assert "from .pricing_interface import" not in reference_source
    assert "double_heston_reference" not in production_source


def test_reference_is_numerically_deterministic_without_runtime_fields(fixture: dict) -> None:
    case = fixture["cases"][10]
    first = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    second = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    assert first == second


def _moneyness_bucket(case: dict[str, Any]) -> tuple[str, str]:
    forward = case["spot"] * np.exp((case["rate"] - case["dividend_yield"]) * case["maturity"])
    ratio = case["strike"] / forward
    if case["option_type"] == CALL_OPTION:
        label = "ITM" if ratio < 0.95 else "ATM" if ratio <= 1.05 else "OTM"
        definition = "call: ITM K/F<0.95, ATM 0.95<=K/F<=1.05, OTM K/F>1.05"
    else:
        label = "ITM" if ratio > 1.05 else "ATM" if ratio >= 0.95 else "OTM"
        definition = "put: ITM K/F>1.05, ATM 0.95<=K/F<=1.05, OTM K/F<0.95"
    return label, definition
