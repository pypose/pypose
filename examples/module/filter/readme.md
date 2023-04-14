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
