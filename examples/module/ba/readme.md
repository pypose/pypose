# Sparse Bundle Adjustment

An example of running bundle adjustment with `pp.optim.LM(..., sparse=True)` on a BAL problem.

The default configuration uses the `trafalgar` `problem-257-65132-pre` dataset and writes:

- an initial PNG snapshot before optimization
- one PNG per LM iteration
- a final PNG snapshot of the optimized point cloud plus camera centers

## Requirements

```bash
python -m pip install -U matplotlib
python -m pip install git+https://github.com/sair-lab/bae.git@0.2
```

Sparse LM currently requires CUDA.

## Run

```bash
python examples/module/ba/bundle_adjustment.py
```


## Notes

- The optimizer configuration is intentionally hard-coded to match the BAE example:
  `TrustRegion(up=2.0, down=0.5**4)`, `pp.optim.solver.PCG(tol=1e-4, maxiter=250)`,
  `reject=30`, `20` iterations, and optimized intrinsics.
- The example saves its PNG snapshots to `./examples/module/ba/save/`.

---

If you use the bundle adjustment, please cite the following paper.

```bibtex
@article{zhan2024bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {IEEE Transactions on Robotics (T-RO)},
  year = {2026},
  url = {https://arxiv.org/abs/2409.12190}
}
```
