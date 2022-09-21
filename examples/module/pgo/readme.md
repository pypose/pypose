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
          -h, --help            show this help message and exit
          --device DEVICE       cuda or cpu
          --damping DAMPING     damping factor
          --radius RADIUS       damping factor
          --save SAVE           location of png files to save
          --dataroot DATAROOT   dataset location downloaded
          --dataname DATANAME   dataset name
          --no-vectorize        to save memory
          --vectorize           to accelerate computation
