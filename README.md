# pypose
PyPose = (GTSAM | G2O) x PyTorch

## 1. For Early Users

1.1 Requirement:
    
[PyTorch 1.11+](https://pytorch.org/get-started/locally/) is required.
    
On Ubuntu, MasOS, or Windows, run:
   
    pip install ninja
    pip install functorch

1.2 Test locally:

    git clone --recursive https://github.com/pypose/pypose.git
    cd pypose && python setup.py develop
    
To rebuild pypose, run `rm -rf build/ *.so` to clean the previous build first.

## 2. For Contributors

2.1 Make sure the above installation is correct. 

2.2 Go to [CONTRIBUTING.md](CONTRIBUTING.md)
