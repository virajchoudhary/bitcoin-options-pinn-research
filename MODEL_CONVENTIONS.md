# Model conventions

The canonical ten-parameter contract in `src/constraints.py` requires positive
structural and state parameters, strict factor-wise Feller gaps, slow-first
ordering, and a joint correlation disk. Its NumPy production engine and
adaptive-quadrature reference retain their original source bytes.

The separable four-shock adapter in
`experiments/nifty_multifactor_v4/literature_exact.py` requires factor-wise
correlations and strict Feller gaps, but does not apply the canonical joint
correlation disk. The Bitcoin-study adapter in
`experiments/btc_multifactor_v1/engine.py` makes the Feller condition optional.
Its normalized price is `C/F` for zero discounting and `x=log(F/K)`.
These validators describe different existing contracts.

The PyTorch Fourier pricer is an exact numerical teacher, not a trained
surrogate. Its callers are responsible for input validity.
The regular variance-PINN is a model architecture. No learned weights are
provided. Its PDE tests instantiate small synthetic networks without
optimization or training.

Only the unsupported optional MLX conversion bridge was removed from the
public PINN copy. This avoids imports of excluded MLX and deep-network sources.
The architecture, features, price formulas, and PDE residual are unchanged.
Apple-specific conversion parity is outside this subset's test scope.

Synthetic tests compare numerical engines, enforce their declared contracts,
check parity and bounds, and differentiate the PDE residual.
They do not measure Bitcoin market fidelity or parameter identifiability.
