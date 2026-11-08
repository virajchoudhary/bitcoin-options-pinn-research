# Double Heston research code

European-option pricing and a regular variance-PINN architecture.
This is a code-only extraction from the reviewed research source.
It is not the complete Bitcoin-paper reproduction package.

## Local checks

Use Python 3.13 on Windows for the pinned CPU validation environment.
The dependencies are installed from their publishers, not bundled here.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest tests
```

The tests use hand-designed synthetic parameters and mathematical regression
constants. They do not fetch market data, load research checkpoints, or train
a model.

## Source map

- [Canonical pricing](src/double_heston.py) implements Gauss-Laguerre pricing.
- [Independent reference](src/double_heston_reference.py) uses adaptive quadrature.
- [Differentiable pricing](src/mentor_dh_pinn/torch_pricer.py) provides PyTorch prices.
- [Regular PINN](src/mentor_dh_pinn/regular_pinn_torch.py) defines the network and PDE residual.
- [Synthetic labels](src/mentor_dh_pinn/regular_pinn_data.py) evaluates model-generated inputs.
- [Four-shock adapter](experiments/nifty_multifactor_v4/literature_exact.py) keeps its separate convention.
- [Bitcoin-study adapter](experiments/btc_multifactor_v1/engine.py) allows an optional Feller condition.
- [Model conventions](MODEL_CONVENTIONS.md) explains why the adapters are not interchangeable.

## Excluded material

No Deribit responses, quotes, forwards, fitted market parameters, prediction
tables, market-derived summaries, figures, datasets, or trained weights are
distributed here. Submitted manuscripts, editorial correspondence, and
proposed manuscript corrections are also excluded.

The original research and its complete Git history remain in separate private
local preservation archives. They are not ancestors of this candidate.
This directory does not make an existing public repository or its history
private.

Full paper replay, saved-C3 scoring, and original model selection are not
available from this public subset. Existing research fidelity failures and
missing C1/C2 checkpoints are not resolved by extracting code.
The source does not establish publication acceptance or a trading strategy.

See [provenance](PROVENANCE.md) and [distribution status](RIGHTS.md).
