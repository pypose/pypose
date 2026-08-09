# Pose Graph Optimization

An example of pose graph optimization (PGO) using either sparse or dense
Levenberg-Marquardt optimization.

## Requirements

```bash
python -m pip install -U matplotlib
```

Sparse optimization is enabled by default and requires CUDA. Use
`--no-sparse --device cpu` to run the dense implementation on CPU.

## Prepare Dataset (Optional)

* The PGO sample data will be downloaded automatically.
* Set dataset `download` option to `False` if you have other data in same format.
  * In this case, the data in `DATAROOT` folder should look like:

        DATAROOT
            ├── parking-garage.g2o

## Run

Run sparse PGO using the default CUDA device:

```bash
python examples/module/pgo/pgo.py
```

This is equivalent to passing `--sparse`. Sparse mode uses the `PCG` solver and
sparse Jacobian assembly. Information matrices are not currently supported by
the sparse optimizer, so `infos` is not passed as a weight in this mode.

Run dense PGO with the dataset information matrices as optimizer weights:

```bash
python examples/module/pgo/pgo.py --no-sparse --device cpu
```

Dense mode uses the `Cholesky` solver and passes `infos` to the optimizer.

To select a CUDA device or a different dataset directory:

```bash
python examples/module/pgo/pgo.py --device cuda:0 --dataroot DATAROOT
```

Replace `DATAROOT` with the directory containing the dataset.

Other supported arguments:

        Pose Graph Optimization

        optional arguments:
          -h, --help           show this help message and exit
          --device DEVICE      cuda or cpu
          --radius RADIUS      trust region radius
          --save SAVE          files location to save
          --dataroot DATAROOT  dataset location
          --dataname DATANAME  dataset name
          --no-sparse          use dense optimization with information matrices
          --sparse             use sparse Jacobians (information matrices are
                               unsupported)
          --no-vectorize       disable vectorization in dense mode to save memory
                               (incompatible with --sparse)
          --vectorize          vectorize dense Jacobian computation when --sparse
                               is False (default in dense mode)

## Notes

- Sparse mode is the default and is recommended for large pose graphs.
- Dense mode incorporates the dataset information matrices but can require
  considerably more memory.
- Sparse Jacobian assembly is always vectorized, so the vectorization options
  do not apply in sparse mode. Dense mode enables `--vectorize` by default; use
  `--no-vectorize` to reduce peak memory usage at the cost of speed.
