# Pose Graph Optimization

An example of pose graph optimization (PGO) using sparse Levenberg-Marquardt
optimization.

## Requirements

```bash
python -m pip install -U matplotlib
```

Sparse optimization requires CUDA.

## Prepare Dataset (Optional)

The PGO samples below are downloaded automatically if the selected `--datafile` is missing.

* [parking-garage.g2o](https://github.com/pypose/bae/releases/download/0.2.2/parking-garage.g2o) (default)
* [sphere_bignoise_vertex3.g2o](https://github.com/pypose/bae/releases/download/0.2.2/sphere_bignoise_vertex3.g2o)
* [sphere_g2o.g2o](https://github.com/pypose/bae/releases/download/0.2.2/sphere_g2o.g2o)
* [torus3D.g2o](https://github.com/pypose/bae/releases/download/0.2.2/torus3D.g2o)
* [grid3D.g2o](https://github.com/pypose/bae/releases/download/0.2.2/grid3D.g2o)

You can also select an existing custom `.g2o` file with `--datafile`.

## Run

Run PGO using the default CUDA device:

```bash
python examples/module/pgo/pgo.py
```

The example uses the `PCG` solver and sparse Jacobian assembly, with information
matrices applied as residual weights.

To run Sphere big-noise, replace `DATAROOT` with your dataset directory:

```bash
python examples/module/pgo/pgo.py --datafile DATAROOT/sphere_bignoise_vertex3.g2o
```

Other supported arguments:

        Pose Graph Optimization

        optional arguments:
          -h, --help           show this help message and exit
          --device DEVICE      cuda device
          --radius RADIUS      initial trust region radius (default: 100)
          --save SAVE          files location to save
          --datafile DATAFILE  dataset file path
          --steps STEPS        maximum optimization steps (default: 80)
---

If you use the pose graph optimization example, please cite the following paper.

```bibtex
@article{zhan2024bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {IEEE Transactions on Robotics (T-RO)},
  year = {2026},
  url = {https://arxiv.org/abs/2409.12190}
}
```
