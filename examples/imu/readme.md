## TO run the IMU_preintegration module:

    pip install wget
    pip install pykitti
    python -m pip install -U matplotlib


## To run the IMU preintegration on Kitti Dataset:

* Download KITTI IMU sample from [here](https://github.com/pypose/IMU_preintegration/releases/download/Kitti/2011_09_26.zip)
* Extract ZIP file to a folder, so that it looks like:

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

* Run:

    python imu_preint.py --dataroot DATAROOT --datadrive 0018 0022

* Other supported arguments:

        IMU Preintegration

        optional arguments:
        -h, --help            show this help message and exit
        --device DEVICE       cuda or cpu
        --batch-size BATCH_SIZE
                                minibatch size
        --save SAVE           location of png files to save
        --dataroot DATAROOT   dataset location downloaded
        --dataname DATANAME   dataset name
        --datadrive DATADRIVE [DATADRIVE ...]
                                data sequences
        --plot3d              plot figure in 3d space, defalue: False
