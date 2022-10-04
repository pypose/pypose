# IMU examples

Examples of IMU [preintegrating](./imu_integrator.py) and IMU [correction](./imu_corrector.py).

## Installation

    python -m pip install -U matplotlib

## Prepare Dataset

* Download KITTI IMU sample data [2011_09_26.zip](https://github.com/pypose/IMU_preintegration/releases/download/Kitti/2011_09_26.zip).
* Extract the file to any folder `DATAROOT`, so that it looks like:

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

        python imu_integrator.py --dataroot DATAROOT --datadrive 0018 0022

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
