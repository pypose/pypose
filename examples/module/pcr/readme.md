# Point Cloud Registration

An example for Point Cloud Registration (PCR)

## Installation

    python -m pip install -U matplotlib

## Run

Iterative Closest Point (ICP)
```bash
python examples/module/pcr/icp.py
```

Or
```bash
python examples/module/pcr/icp.py --device cpu --steps 200 --patience 5 --decreasing 1e-5 --verbose --show
```

* Other supported arguments:

    ICP

        optional arguments:
        -h, --help                  show this help message and exit
        --device DEVICE             cuda or cpu
        --steps STEPS               maximum number of ICP will step, default: 200
        --patience PATIENCE         number of steps with no loss 'decreasing' is seen, default: 5
        --decreasing DECREASING     relative loss decreasing used to count the number of patience steps, default: 1e-3
        --verbose                   print a message for each step, default: False
        --save SAVE                 location of png files to save
        --show                      show plot, default: False
