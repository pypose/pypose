# Point Cloud Registration

An example for Point Cloud Registration (PCR)

## Installation

    python -m pip install -U matplotlib
    pip install torchvision

## Run

Iterative Closest Point (ICP)
```bash
python examples/module/pcr/icp.py
```

Or
```bash
python examples/module/pcr/icp.py --verbose --show --device cpu --steps 100
```

* Other supported arguments:

    ICP

        optional arguments:
        -h, --help              show this help message and exit
        --device DEVICE         cuda or cpu
        --steps STEPS           maximum number of ICP will step, default: 200
        --patience PATIENCE     steps with no loss 'decreasing' is seen, default: 5
        --decreasing DECREASING relative loss decreasing used to count the number of patience steps, default: 1e-3
        --verbose               print a message for each step, default: False
        --dataroot DATAROOT     dataset location
        --save SAVE             location of png files to save
        --show                  show plot, default: False
