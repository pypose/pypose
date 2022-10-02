# pypose

#### To connect classic robotics with modern learning methods.

-----

### Current Features

##### [LieTensor](https://pypose.org/docs/main/modules/)

- Lie Group: [`SO3`](https://pypose.org/docs/main/generated/pypose.SO3/), [`SE3`](https://pypose.org/docs/main/generated/pypose.SE3/), [`Sim3`](https://pypose.org/docs/main/generated/pypose.Sim3/), [`RxSO3`](https://pypose.org/docs/main/generated/pypose.RxSO3/)
- Lie Algebra: [`so3`](https://pypose.org/docs/main/generated/pypose.so3/), [`se3`](https://pypose.org/docs/main/generated/pypose.se3/), [`sim3`](https://pypose.org/docs/main/generated/pypose.sim3/), [`rxso3`](https://pypose.org/docs/main/generated/pypose.rxso3/)

##### [Modules](https://pypose.org/docs/main/modules/)

- [`System`](https://pypose.org/docs/main/generated/pypose.module.System)
- [`IMUPreintegration`](https://pypose.org/docs/main/generated/pypose.module.IMUPreintegrator/)
- ......

##### [Second-order Optimizers](https://pypose.org/docs/main/optim/)

- [`GaussNewton`](https://pypose.org/docs/main/generated/pypose.optim.GaussNewton)
- [`LevenbergMarquardt`](https://pypose.org/docs/main/generated/pypose.optim.LevenbergMarquardt/)
- ......

##### Efficiency-based design

- We support parallel computing for Jacobian of LieTensor.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/8695500/193468407-acbadb86-15d9-45d3-b7ef-864db744df38.png">

Efficiency comparison of Lie group operations on CPU and GPU (we take Theseus performance as 1×).

More information about efficiency comparison goes to the paper.

## Getting Started
    
### Installing

#### Install from **pypi**
```bash
pip install pypose
```

#### **From source**
```bash
git clone https://github.com/pypose/pypose.git && cd pypose
python setup.py develop
```

#### For Early Users

1. Requirement:

On Ubuntu, MasOS, or Windows, install [PyTorch](https://pytorch.org/), then run:

```bash
pip install -r requirements/main.txt
```

2.  Install locally:

```bash
git clone  https://github.com/pypose/pypose.git
cd pypose && python setup.py develop
```

3. Run Test

```bash
pytest
```

####  For Contributors

1. Make sure the above installation is correct. 

2. Go to [CONTRIBUTING.md](CONTRIBUTING.md)

## Citing PyPose

If you use PyPose, please cite the paper below.

```bibtex
@article{wang2022pypose,
  title   = {{PyPose: A Library for Robot Learning with Physics-based Optimization}},
  author  = {Chen Wang, Dasong Gao, Kuan Xu, Junyi Geng1, Yaoyu Hu, Yuheng Qiu, Bowen Li, Fan Yang, Brady Moon, Abhinav Pandey, Aryan, Jiahe Xu, Tianhao Wu, Haonan He, Daning Huang, Zhongqiang Ren, Shibo Zhao, Taimeng Fu, Pranay Reddy, Xiao Lin, Wenshan Wang, Jingnan Shi, Rajat Talak, Han Wang, Huai Yu, Shanzhao Wang, Ananth Kashyap, Rohan Bandaru, Karthik Dantu, Jiajun Wu, Luca Carlone, Marco Hutter, Sebastian Scherer},
  journal = {arXiv},
  year    = {2022}
}
```
