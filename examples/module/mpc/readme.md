# Model Predictive Control

Two Learning Examples for Model Predictive Control (MPC)

## Installation

    python -m pip install -U matplotlib

## Run:

Learning example, linear time-invariant system
```bash
python examples/module/mpc/linear.py
```

Cart-Pole learning example, nonlinear time-invariant system
```bash
python examples/module/mpc/cartpole.py
```

* Other supported arguments:

    -h, --help       show this help message and exit
    --device DEVICE  cuda or cpu
    --save SAVE      location of png files to save
    --show           show plot, default: False
