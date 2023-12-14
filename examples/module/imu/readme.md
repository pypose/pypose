# IMU examples

Examples of IMU [preintegrating](./imu_integrator.py) and IMU [correction](./imu_corrector.py).

## Installation

    python -m pip install -U matplotlib
    pip install opencv-python
    pip install pykitti tqdm
    pip install torchvision

## Prepare Dataset (Optional)

* The KITTI IMU sample data will be downloaded automatically.
* Set dataset `download` option to `False` if you have other data in same format.
  * In this case, the data in `DATAROOT` folder should look like:

        DATAROOT
            ├── 2011_09_26
                ├── 2011_09_26_drive_0001_sync
                ├── 2011_09_26_drive_0002_sync
                ├── 2011_09_26_drive_0005_sync
                .
                .
                .
                ├── 2011_09_26_drive_0018_sync
                ├── 2011_09_26_drive_0020_sync
                ├── 2011_09_26_drive_0022_sync
                ├── calib_cam_to_cam.txt
                ├── calib_imu_to_velo.txt
                └── calib_velo_to_cam.txt

# IMU Preintegration:

```bash
python examples/module/imu/imu_integrator.py
```

Or

```bash
python examples/module/imu/imu_integrator.py --datadrive 0018 0022 --dataroot DATAROOT
```

Note: change `DATAROOT` to the folder you select.

* Data Drive to select:

        0001 0002 0005 0009 0011 0013 0014 0015 0017 0018 0019 0020 0022 0005

* Other supported arguments:

        IMU Preintegration

        optional arguments:
          -h, --help            show this help message and exit
          --device DEVICE       cuda or cpu
          --integrating-step INTEGRATING_STEP
                                number of integrated steps
          --batch-size BATCH_SIZE
                                batch size, only support 1 now
          --save SAVE           location of png files to save
          --dataroot DATAROOT   dataset location downloaded
          --dataname DATANAME   dataset name
          --datadrive DATADRIVE [DATADRIVE ...]
                                data sequences
          --plot3d              plot in 3D space, default: False

# IMU Correction:

        python imu_integrator.py --dataroot DATAROOT

Note: change `DATAROOT` to the folder you select.

* Data Drive to select:

        0001 0002 0005 0009 0011 0013 0014 0015 0017 0018 0019 0020 0022 0005

* Other supported arguments:

        optional arguments:
        -h, --help            show this help message and exit
        --device DEVICE       cuda or cpu
        --batch-size BATCH_SIZE
                                batch size
        --max_epoches MAX_EPOCHES
                                max_epoches
        --dataroot DATAROOT   dataset location downloaded
        --dataname DATANAME   dataset name
        --datadrive DATADRIVE [DATADRIVE ...]
                                data sequences
        --load_ckpt

![IMUcorrector](https://github.com/pypose/pypose/assets/22726519/fab23d94-58ce-4948-98f0-c4d5e66e6080)

