# pypose
PyPose = (GTSAM | G2O) x PyTorch

To connect classic robotics with modern learning methods.

## 1. For Early Users

1.1 Requirement:

On Ubuntu, MasOS, or Windows, run:

    pip install -r requirements.txt

1.2 Install locally:

    git clone --recursive https://github.com/pypose/pypose.git
    cd pypose && python setup.py develop

1.3 Run Test

    pytest

To rebuild pypose, run `rm -rf build/ *.so` to clean the previous build first.

## 2. For Contributors

2.1 Make sure the above installation is correct. 

2.2 Go to [CONTRIBUTING.md](CONTRIBUTING.md)
