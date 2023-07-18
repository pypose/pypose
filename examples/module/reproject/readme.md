# Reprojection Error Optimization

An example for reprojection error optimization between adjacent frames

## Installation

  * matplotlib
  * opencv-python


## Run:

```bash
python examples/module/reproject/reproject.py
```

Or

```bash
python examples/module/reproject/reproject.py --device cuda --dataroot DATAROOT
```

Note: change `DATAROOT` to the folder you select.

* Other supported arguments:

        Pose Graph Optimization

        optional arguments:
          -h, --help           show this help message and exit
          --device DEVICE      cuda or cpu (default: cuda)
          --save SAVE          files location to save (default: ./est_traj.txt)
          --dataroot DATAROOT  dataset location
          --vectorize          to accelerate computation
