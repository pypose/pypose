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


## Notes

- The optimizer configuration is intentionally hard-coded to match the BAE example:
  `TrustRegion(up=2.0, down=0.5**4)`, `pp.optim.solver.PCG(tol=1e-4, maxiter=250)`,
  `reject=30`, `20` iterations, and optimized intrinsics.
- The example saves its GIF and final PNG to `./examples/module/bundle_adjustment/save/`.

---

If you use the bundle adjustment, please cite the following paper.

```bibtex
@article{zhan2024bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {arXiv preprint arXiv:2409.12190},
  year = {2024},
  url = {https://arxiv.org/abs/2409.12190}
}
```
