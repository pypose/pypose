# Filter examples


## Installation

    python -m pip install -U matplotlib

# Run

Extended Kalman Filter (EKF)
```bash
python examples/module/filter/ekf.py
```

Unscented Kalman Filter (UKF)

```bash
python examples/module/filter/ukf.py
```

Particle Filter (PF)

```bash
python examples/module/filter/pf.py
```

## Batched comparison

Run EKF, UKF and PF on the same fixed-seed bicycle trajectories:

```bash
python examples/module/filter/batched.py --device cpu
python examples/module/filter/batched.py --device cuda --shared-covariance
```

States and observations are `(B, T, 3)`, inputs are `(B, T, 2)` and
estimation covariances are `(B, T, 3, 3)`. Batch entries are independent systems;
time is a separate dimension, and the filters manage their own sigma-point or
particle dimension. Each observation measures the next state **after** process
noise has been added. All three filters receive the same measurements and
initial estimates.

The default uses different process and measurement covariances per system;
`--shared-covariance` uses common `(3, 3)` matrices. The example prints position
and wrapped-heading RMSE separately, and saves one plot per filter and trajectory.
Use `--batch`, `--steps`, `--particles`, `--save` and `--show` to adjust the run.

* Other supported arguments:

    EKF

        optional arguments:
        -h, --help       show this help message and exit
        --device DEVICE  cuda or cpu
        --save SAVE      location of png files to save
        --show           show plot, default: False

    UKF

        optional arguments:
        -h, --help       show this help message and exit
        --device DEVICE  cuda or cpu
        --k K            An integer parameter for weighting the sigma points.
        --save SAVE      location of png files to save
        --show           show plot, default: False

    PF

        optional arguments:
        -h, --help       show this help message and exit
        --device DEVICE  cuda or cpu
        --N N            The number of particle
        --save SAVE      location of png files to save
        --show           show plot, default: False
