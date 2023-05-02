# Point Cloud Registration

An example for Point Cloud Registration (PCR)

## Installation

    python -m pip install -U matplotlib

## Run

Iterative Closest Point (ICP)
```bash
python examples/module/pcr/icp.py
```

* Other supported arguments:

    ICP

        optional arguments:
        -h, --help       show this help message and exit
        --device DEVICE  cuda or cpu
        --save SAVE      location of png files to save
        --show           show plot, default: False
