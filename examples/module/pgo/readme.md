# Pose Graph Optimization

An example for Pose Graph Optimization (PGO)

## Installation

    python -m pip install -U matplotlib

## Prepare Dataset

* Download the Parking Garage G2O sample data [parking-garage.g2o](https://www.dropbox.com/s/zu23p8d522qccor/parking-garage.g2o?dl=0).
* Extract the file to any folder `DATAROOT`, so that it looks like:

        DATAROOT
            ├── parking-garage.g2o

## Run:

        python pgo.py --dataroot DATAROOT --device cuda:0

Note: change `DATAROOT` to the folder you select.


* Other supported arguments:

        Pose Graph Optimization

        optional arguments:
          -h, --help           show this help message and exit
          --device DEVICE      cuda or cpu
          --radius RADIUS      trust region radius
          --save SAVE          files location to save
          --dataroot DATAROOT  dataset location
          --dataname DATANAME  dataset name
          --no-vectorize       to save memory
          --vectorize          to accelerate computation

## Note

The current implementation of PGO is not using sparse matrices. Therefore, when the number of poses is very big, the memory consumption can be large (`--vectorize`) or running speed can be slow (`--no-vectorize`). The sparse matrices will be integrated in the next main release, while the API will be mostly unchanged, only internal logic will be updated.
