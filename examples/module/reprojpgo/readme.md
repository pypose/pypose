# Reprojection Error Pose Graph Optimization

![Reprojerr_Visualization](https://github.com/pypose/pypose/assets/47029019/e5c2ab45-7578-4901-a779-9b7f44cce338)

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
  ```
    Pose Graph Optimization

    optional arguments:
      -h, --help           show this help message and exit
      --dataroot DATAROOT  dataset location (default: ./data/Reprojerr_Example)
      --device DEVICE      cuda or cpu (default: cuda)
      --vectorize          to accelerate computation (will use more RAM)
      --dnoise DEPTH_NOISE noise level on point depth (default: 0.1)
      --pnoise POSE_NOISE  noise level on initial pose (default: 0.1)
  ```
