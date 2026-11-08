import io
import math

import numpy as np
import pytest
import torch

from experiments.btc_multifactor_v1 import engine as bitcoin
from experiments.nifty_multifactor_v4 import literature_exact as four_shock
from src.double_heston import price_double_heston_call
from src.mentor_dh_pinn.regular_pinn_data import draw_points, teacher_labels
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN
from tests.synthetic_fixtures import BENCHMARK


@pytest.mark.parametrize("case", BENCHMARK["cases"], ids=lambda case: case["case_id"])
def test_adapters_match_canonical_call_on_synthetic_inputs(case):
    x = math.log(case["spot"] / case["strike"])
    tau = case["maturity"]
    parameters = case["parameters"]
    expected = price_double_heston_call(
        case["spot"], case["strike"], tau, 0.0, 0.0, parameters, node_count=128
    ) / case["spot"]
    tolerance = 2e-5 / case["spot"] + 2e-6 * abs(expected)
    actual = bitcoin.exact(parameters, np.array([x]), np.array([tau]), True)[0]
    other = four_shock.exact(parameters, np.array([x]), np.array([tau]))[0]
    assert actual == pytest.approx(expected, abs=tolerance, rel=0.0)
    assert other == pytest.approx(expected, abs=tolerance, rel=0.0)


def test_optional_feller_contract_remains_distinct():
    parameters = [0.8, 0.04, 0.4, -0.45, 0.05, 3.0, 0.02, 0.25, -0.25, 0.025]
    bitcoin.admissible(parameters, False)
    with pytest.raises(ValueError, match="Feller"):
        bitcoin.admissible(parameters, True)
    with pytest.raises(ValueError, match="Feller"):
        four_shock.admissible(parameters)
    price = bitcoin.exact(parameters, np.array([0.0]), np.array([0.5]), False)
    assert np.isfinite(price).all() and (price >= 0).all() and (price <= 1).all()


@pytest.mark.parametrize("factors", [1, 2])
def test_synthetic_label_execution_without_datasets(factors):
    points = draw_points(4, factors, seed=101)
    labels = teacher_labels(points, factors, gradients=True, full_gradients=True)
    assert labels["q"].shape == (4, 2 + 5 * factors)
    assert np.isfinite(labels["price"]).all()
    assert np.isfinite(labels["dg_dq"][labels["usable"]]).all()
    assert labels["usable"].any()


def test_synthetic_state_dict_round_trip_in_memory_without_training():
    torch.manual_seed(601)
    model = TorchRegularVariancePINN(factors=2, width=10, depth=2)
    coords = torch.tensor([[0.04, 0.05, 0.04, 0.25]], dtype=torch.float64)
    structural = torch.tensor(
        [[0.8, 0.04, 0.19, -0.5], [4.5, 0.05, 0.35, -0.3]], dtype=torch.float64
    )
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = TorchRegularVariancePINN(factors=2, width=10, depth=2)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(model.price(coords, structural), restored.price(coords, structural),
                               atol=0.0, rtol=0.0)
