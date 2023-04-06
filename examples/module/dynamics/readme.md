# Defining Dynamical Systems

Examples for defining general discrete-time nonlinear dynamical systems

## Installation

    python -m pip install -U matplotlib

## Run:

    + The classical Cart-Pole example, nonlinear time-invariant system

        python examples/module/dynamics/cartpole.py

    + A Floquet system example, nonlinear time-varying system

        python examples/module/dynamics/floquet.py

    + Dynamics defined using a neural network, nonlinear time-invariant system

        python examples/module/dynamics/neuralnet.py

    + optional arguments:

        -h, --help       show this help message and exit
        --device DEVICE  cuda or cpu
        --save SAVE      location of png files to save
        --show           show plot, default: False