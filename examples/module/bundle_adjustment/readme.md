# Sparse Bundle Adjustment

An example of running bundle adjustment with `pp.optim.LM(..., sparse=True)` on a BAL problem.

The default configuration uses the `trafalgar` `problem-257-65132-pre` dataset and writes:

- a per-iteration GIF showing the initial reconstruction on the left and the current reconstruction on the right
- a final PNG snapshot of the optimized point cloud plus camera centers

## Requirements

```bash
python -m pip install -U matplotlib
python -m pip install git+https://github.com/zitongzhan/bae.git
```

Sparse LM currently requires CUDA.

## Run

```bash
python examples/module/bundle_adjustment/bundle_adjustment.py
```

Or reuse BAE's BAL loader code directly from GitHub:

```bash
python examples/module/bundle_adjustment/bundle_adjustment.py \
  --loader-source bae
```

## Notes

- The optimizer configuration is intentionally hard-coded to match the BAE example:
  `TrustRegion(up=2.0, down=0.5**4)`, `PCG(tol=1e-4, maxiter=250)`, `reject=30`, `20` iterations, and optimized intrinsics.
- `--loader-source local` is the default. It downloads the BAL problem file and parses it with a local helper in this example.
- `--loader-source bae` is an opt-in mode that fetches `bal_loader.py` and `bal_io.py` from the BAE `release` branch at runtime and imports them from a local cache.
- The example saves its GIF and final PNG to `./examples/module/bundle_adjustment/save/`.
