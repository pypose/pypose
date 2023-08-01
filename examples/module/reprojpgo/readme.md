# Reprojection Error Pose Graph Optimization

![Reprojerr_Visualization](https://user-images.githubusercontent.com/47029019/257399825-2f47931c-adc8-494e-99c6-d0c2c5d7c306.gif)

An example for pose graph optimization between adjacent frames using cropped trajectory from [the TartanAir dataset](https://theairlab.org/tartanair-dataset/) (Easy, abandoned factory, sequence P001).

## Installation

  * matplotlib
  * opencv-python


## Run

```bash
python examples/module/reprojpgo/reprojpgo.py
```

Or

```bash
python examples/module/reprojpgo/reprojpgo.py --device cuda --vectorize
```

* Other supported arguments:

        Pose Graph Optimization

        optional arguments:
          -h, --help           show this help message and exit
          --device DEVICE      cuda or cpu (default: cuda)
          --dataroot DATAROOT  dataset location (default: ./data/Reprojerr_Example)
          --vectorize          to accelerate computation (will use more RAM)
