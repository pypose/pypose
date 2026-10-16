# Pose Graph Optimization

An example of pose graph optimization (PGO) using sparse Levenberg-Marquardt
optimization.

## Requirements

```bash
python -m pip install -U matplotlib
```

Sparse optimization requires CUDA.

## Prepare Dataset (Optional)

* The PGO sample data will be downloaded automatically.
* Set dataset `download` option to `False` if you have other data in same format.
  * In this case, the data in `DATAROOT` folder should look like:

        DATAROOT
            ├── parking-garage.g2o

## Run

Run PGO using the default CUDA device:

```bash
python examples/module/pgo/pgo.py
```

The example uses the `PCG` solver and sparse Jacobian assembly. Information
matrices (`infos`) are loaded with the dataset but are not currently supported
as weights by the sparse optimizer. Support will be added in a future release.

To select a CUDA device or a different dataset directory:

```bash
python examples/module/pgo/pgo.py --device cuda:0 --dataroot DATAROOT
```

Replace `DATAROOT` with the directory containing the dataset.

Other supported arguments:

        Pose Graph Optimization

        optional arguments:
          -h, --help           show this help message and exit
          --device DEVICE      cuda device
          --radius RADIUS      trust region radius
          --save SAVE          files location to save
          --dataroot DATAROOT  dataset location
          --dataname DATANAME  dataset name
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
