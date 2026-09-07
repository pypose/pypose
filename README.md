## PyPose: A Library for Robot Learning with Physics-based Optimization

![robot](https://user-images.githubusercontent.com/8695500/193484553-2da66824-4461-4aca-ad8c-b17c05bef067.png)

-----

Deep learning has had remarkable success in robotic perception, but its data-centric nature suffers when it comes to generalizing to ever-changing environments. By contrast, physics-based optimization generalizes better, but it does not perform as well in complicated tasks due to the lack of high-level semantic information and the reliance on manual parametric tuning. To take advantage of these two complementary worlds, we present PyPose: a **robotics-oriented**, **PyTorch-based** library that combines **deep perceptual models** with **physics-based optimization techniques**. Our design goal for PyPose is to make it **user-friendly**, **efficient**, and **interpretable** with a tidy and well-organized architecture. Using an **imperative style interface**, it can be easily integrated into **real-world robotic applications**.


-----

### Current Features

##### [LieTensor](https://pypose.org/docs/main/modules/)

- Lie group: [`SO3`](https://pypose.org/docs/main/generated/pypose.SO3/), [`SE3`](https://pypose.org/docs/main/generated/pypose.SE3/), [`Sim3`](https://pypose.org/docs/main/generated/pypose.Sim3/), [`RxSO3`](https://pypose.org/docs/main/generated/pypose.RxSO3/)
- Lie algebra: [`so3`](https://pypose.org/docs/main/generated/pypose.so3/), [`se3`](https://pypose.org/docs/main/generated/pypose.se3/), [`sim3`](https://pypose.org/docs/main/generated/pypose.sim3/), [`rxso3`](https://pypose.org/docs/main/generated/pypose.rxso3/)

##### [Modules](https://pypose.org/docs/main/modules/)

- System: [`LTI`](https://pypose.org/docs/main/generated/pypose.module.LTI), [`LTV`](https://pypose.org/docs/main/generated/pypose.module.LTV), [`NLS`](https://pypose.org/docs/main/generated/pypose.module.NLS)
- Filter: [`EKF`](https://pypose.org/docs/main/generated/pypose.module.EKF/), [`UKF`](https://pypose.org/docs/main/generated/pypose.module.UKF/), [`PF`](https://pypose.org/docs/main/generated/pypose.module.PF/)
- PnP Solver: [`EPnP`](https://pypose.org/docs/main/generated/pypose.module.EPnP/)
- Linear Quadratic Regulator: [`LQR`](https://pypose.org/docs/main/generated/pypose.module.LQR/)
- IMU Preintegration: [`IMUPreintegrator`](https://pypose.org/docs/main/generated/pypose.module.IMUPreintegrator/)
- ......

##### [Second-order Optimizers](https://pypose.org/docs/main/optim/)

- [`GaussNewton`](https://pypose.org/docs/main/generated/pypose.optim.GaussNewton)
- [`LevenbergMarquardt`](https://pypose.org/docs/main/generated/pypose.optim.LevenbergMarquardt/)
- ......

Want more features? [Create an issue here](https://github.com/pypose/pypose/issues) to request new features.

##### PyPose is highly efficient and supports parallel computing for Jacobian of Lie group and Lie algebra. See following comparison.

<img width="1167" alt="image" src="https://user-images.githubusercontent.com/8695500/203210668-1a90224a-ae08-4d31-b9d1-e293be75ef3e.png">

Efficiency and memory comparison of batched Lie group operations (we take Theseus performance as 1×).

More information about efficiency comparison goes to [our paper for PyPose](https://arxiv.org/abs/2209.15428).

## Getting Started

### Installation

#### Install from **pypi**
```bash
pip install pypose
```

#### Install from source

1. Requirement:

On Ubuntu, macOS, or Windows, install [PyTorch](https://pytorch.org/), then run:

```bash
pip install -r requirements/runtime.txt
```

2. Install locally:

```bash
git clone  https://github.com/pypose/pypose.git
cd pypose && pip install -e .
```

3. Run tests

```bash
pytest
```

####  For contributors

1. Make sure the above installation is correct.

2. Go to [CONTRIBUTING.md](CONTRIBUTING.md)


#### Examples

1. The following code sample shows how to rotate random points and compute the gradient of batched rotation.

```python
>>> import torch, pypose as pp

>>> # A random so(3) LieTensor
>>> r = pp.randn_so3(2, requires_grad=True)
    so3Type LieTensor:
    tensor([[ 0.1606,  0.0232, -1.5516],
            [-0.0807, -0.7184, -0.1102]], requires_grad=True)

>>> R = r.Exp() # Equivalent to: R = pp.Exp(r)
    SO3Type LieTensor:
    tensor([[ 0.0724,  0.0104, -0.6995,  0.7109],
            [-0.0395, -0.3513, -0.0539,  0.9339]], grad_fn=<AliasBackward0>)

>>> p = R @ torch.randn(3) # Rotate random point
    tensor([[ 0.8045, -0.8555,  0.5260],
            [ 0.3502,  0.8337,  0.9154]], grad_fn=<ViewBackward0>)

>>> p.sum().backward()     # Compute gradient
>>> r.grad                 # Print gradient
    tensor([[-0.7920, -0.9510,  1.7110],
            [-0.2659,  0.5709, -0.3855]])
```

2. This example shows how to estimate batched inverse of transform by a second-order optimizer. Two usage options for a `scheduler` are provided, each of which can work independently.

```python
>>> from torch import nn
>>> import torch, pypose as pp
>>> from pypose.optim import LM
>>> from pypose.optim.strategy import Constant
>>> from pypose.optim.scheduler import StopOnPlateau

>>> class InvNet(nn.Module):
...
...     def __init__(self, *dim):
...         super().__init__()
...         init = pp.randn_SE3(*dim)
...         self.pose = pp.Parameter(init)
...
...     def forward(self, input):
...         error = (self.pose @ input).Log()
...         return error.tensor()

>>> device = torch.device("cuda")
>>> input = pp.randn_SE3(2, 2, device=device)
>>> invnet = InvNet(2, 2).to(device)
>>> strategy = Constant(damping=1e-4)
>>> optimizer = LM(invnet, strategy=strategy)
>>> scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

>>> # 1st option, full optimization
>>> scheduler.optimize(input=input)

>>> # 2nd option, step optimization
>>> while scheduler.continual():
...     loss = optimizer.step(input)
...     scheduler.step(loss)

>>> # Note: remove one of the above options for usage!
```

3. May 2026: Starting from v0.9.5, PyPose introduces sparse Jacobian tracing, enabling
efficient sparse 2nd-order optimization, significantly accelerating applications such as
bundle adjustment.

```python
>>> import torch
>>> import pypose as pp
>>> from torch import nn
>>> from pypose.optim import LM
>>> from pypose.optim.solver import PCG
>>> from pypose.optim.strategy import TrustRegion
>>> from pypose.optim.scheduler import StopOnPlateau
>>> from pypose.autograd.function import psjac

>>> class ReprojErr(nn.Module):
...     def __init__(self, poses, points):
...         super().__init__()
...         # sjac: enabling tracing of sparse Jacobian
...         self.poses = pp.Parameter(poses, sjac=True)
...         self.points = pp.Parameter(points, sjac=True)
...
...     @psjac  # parallelize assembly of sparse Jacobian
...     def project(poses, points):
...         points = poses.Act(points)
...         return - points[..., :2] / points[..., [2]]
...
...     def forward(self, pixels, cidx, pidx):
...         poses = self.poses[cidx]
...         points = self.points[pidx]
...         return ReprojErr.project(poses, points) - pixels

>>> torch.set_default_device("cuda")
>>> npts, poses = 8, pp.randn_SE3(1)
>>> points = torch.randn(npts, 3)
>>> points[:, 2] += 4  # positive depth
>>> cidx = torch.zeros(npts, dtype=torch.long)
>>> pidx = torch.arange(npts)
>>> pixels = torch.randn(npts, 2)
>>> inputs = (pixels, cidx, pidx)

>>> model = ReprojErr(poses, points)
>>> solver = PCG(tol=1e-4, maxiter=250)
>>> strategy = TrustRegion(up=2.0, down=0.5**4)
>>> optimizer = LM(model, solver, strategy, sparse=True)
>>> scheduler = StopOnPlateau(optimizer, steps=5, verbose=True)

>>> while scheduler.continual():
...     loss = optimizer.step(inputs)
...     scheduler.step(loss)
```

For more usage, see [Documentation](https://pypose.org/docs). For more applications, see [Examples](https://github.com/pypose/pypose/tree/main/examples).

## Citing PyPose

If you use PyPose, please cite the paper below. You may also [download it here](https://arxiv.org/abs/2209.15428).

```bibtex
@inproceedings{wang2023pypose,
  title = {{PyPose}: A Library for Robot Learning with Physics-based Optimization},
  author = {Wang, Chen and Gao, Dasong and Xu, Kuan and Geng, Junyi and Hu, Yaoyu and Qiu, Yuheng and Li, Bowen and Yang, Fan and Moon, Brady and Pandey, Abhinav and Aryan and Xu, Jiahe and Wu, Tianhao and He, Haonan and Huang, Daning and Ren, Zhongqiang and Zhao, Shibo and Fu, Taimeng and Reddy, Pranay and Lin, Xiao and Wang, Wenshan and Shi, Jingnan and Talak, Rajat and Cao, Kun and Du, Yi and Wang, Han and Yu, Huai and Wang, Shanzhao and Chen, Siyu and Kashyap, Ananth  and Bandaru, Rohan and Dantu, Karthik and Wu, Jiajun and Xie, Lihua and Carlone, Luca and Hutter, Marco and Scherer, Sebastian},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```

If you use the sparse Jacobian, GPU sparse linear algebra, conjugate gradient solver, or sparse LM optimizer in PyPose, please cite the [following paper](https://arxiv.org/abs/2409.12190).

```bibtex
@article{zhan2026bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {IEEE Transactions on Robotics (T-RO)},
  year = {2026},
  url = {https://arxiv.org/abs/2409.12190}
}
```

More papers describing PyPose:

```bibtex
@inproceedings{zhan2023pypose,
  title = {{PyPose} v0.6: The Imperative Programming Interface for Robotics},
  author = {Zitong Zhan and Xiangfu Li and Qihang Li and Haonan He and Abhinav Pandey and Haitao Xiao and Yangmengfei Xu and Xiangyu Chen and Kuan Xu and Kun Cao and Zhipeng Zhao and Zihan Wang and Huan Xu and Zihang Fang and Yutian Chen and Wentao Wang and Xu Fang and Yi Du and Tianhao Wu and Xiao Lin and Yuheng Qiu and Fan Yang and Jingnan Shi and Shaoshu Su and Yiren Lu and Taimeng Fu and Karthik Dantu and Jiajun Wu and Lihua Xie and Marco Hutter and Luca Carlone and Sebastian Scherer and Daning Huang and Yaoyu Hu and Junyi Geng and Chen Wang},
  year = {2023},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) Workshop},
}
```


## 🌐 Web Resources & Interactive Index
- [RUN N SHOOT](https://eduquestsjp.pages.dev/run-n-shoot.html)
- [FLUFFY MANIA](https://eduquestses.pages.dev/fluffy-mania.html)
- [BUILD AND RUN](https://welearnaction.onrender.com/build-and-run.html)
- [CATEGORY SKILL254](https://eduquestkr.pages.dev/category-skill254.html)
- [IDLE POP MERGE](https://eduquests.onrender.com/idle-pop-merge.html)
- [TRAVEL STORY MATCH](https://brainquests.pages.dev/travel-story-match.html)
- [INDEX42](https://eduquests.pages.dev/index42.html)
- [POP ADVENTURE](https://ieduquests.web.app/pop-adventure.html)
- [BUBBLE SHOOTER NEON](https://ieduquests.web.app/bubble-shooter-neon.html)
- [SKILLFUL FINGER](https://eduquestses.pages.dev/skillful-finger.html)
- [CATEGORY MANAGEMENT210](https://brainquests.pages.dev/category-management210.html)
- [CATEGORY UNBLOCKEDGAMES](https://brainquests.pages.dev/category-unblockedgames.html)
- [INDEX16](https://brainquests.pages.dev/index16.html)
- [MOW IT](https://eduquestspt.pages.dev/mow-it.html)
- [CYBER ARROW](https://eduquestspt.pages.dev/cyber-arrow.html)
- [BURGER EMPIRE](https://eduquestkr.pages.dev/burger-empire.html)
- [OPENGUESSR](https://eduquests.pages.dev/openguessr.html)
- [OBBY WITH FRIENDS DRAW AND JUMP](https://ieduquests.web.app/obby-with-friends-draw-and-jump.html)
- [ZOMBIE DERBY PIXEL SURVIVAL](https://brainquests.pages.dev/zombie-derby-pixel-survival.html)
- [FLOWER BLOCK](https://brainquests.pages.dev/flower-block.html)
- [CATEGORY STICKMAN 2](https://brainquests.pages.dev/category-stickman-2.html)
- [FISH OUT OF WATER](https://brainquests.pages.dev/fish-out-of-water.html)
- [MIRROR SHAPE](https://eduquestses.pages.dev/mirror-shape.html)
- [IDLE BATHROOM EMPIRE TYCOON](https://eduquests.pages.dev/idle-bathroom-empire-tycoon.html)
- [SUPER DOG HERO DASH](https://eduquestses.pages.dev/super-dog-hero-dash.html)
- [SPIDER EVOLUTION](https://brainquests.pages.dev/spider-evolution.html)
- [CATEGORY ANIMAL216](https://eduquestkr.pages.dev/category-animal216.html)
- [HEXAGON](https://brainquests.pages.dev/hexagon.html)
- [STEAL BRAINROT MONSTERS](https://brainquests.pages.dev/steal-brainrot-monsters.html)
- [ONLINE PORTAL](https://themindplays.pages.dev/)
- [WORD ART COLOR BOOK PUZZLE](https://eduquestses.pages.dev/word-art-color-book-puzzle.html)
- [CUTE CATS ADVENTURES](https://brainquests.pages.dev/cute-cats-adventures.html)
- [2 PLAYER BATTLE](https://eduquestses.pages.dev/2-player-battle.html)
- [SITEMAP](https://brainquests.onrender.com/sitemap.html)
- [BOWMASTERS](https://brainquests.pages.dev/bowmasters.html)
- [COIN MERGE](https://eduquests.pages.dev/coin-merge.html)
- [BLACK JUMP](https://eduquestspt.pages.dev/black-jump.html)
- [DOT BY DOT](https://brainquests.pages.dev/dot-by-dot.html)
- [SCREW IT OUT JAM MATCHING COLORED SCREWS](https://brainquests.pages.dev/screw-it-out-jam-matching-colored-screws.html)
- [CATEGORY SURVIVAL](https://brainquests.pages.dev/category-survival.html)
- [BATTLE TANKS FIRESTORM](https://ieduquests.web.app/battle-tanks-firestorm.html)
- [FROGGY HOP](https://eduquestkr.pages.dev/froggy-hop.html)
- [EMERGENCY JAM](https://eduquestspt.pages.dev/emergency-jam.html)
- [LINK FLOW](https://brainquests.pages.dev/link-flow.html)
- [SANTA GO](https://ieduquests.web.app/santa-go.html)
- [ROBYBOX SPACE STATION WAREHOUSE](https://brainquests.pages.dev/robybox-space-station-warehouse.html)
- [CRAZY BIKE STUNTS PVP](https://ieduquests.web.app/crazy-bike-stunts-pvp.html)
- [CATEGORY COOKING](https://brainquests.pages.dev/category-cooking.html)
- [CATEGORY BATTLE](https://eduquestspt.pages.dev/category-battle.html)
- [CANDY CASCADE](https://brainquests.pages.dev/candy-cascade.html)
- [CATEGORY CRAFTING45](https://eduquestkr.pages.dev/category-crafting45.html)
- [ANIMAL LINK](https://ieduquests.web.app/animal-link.html)
- [10K](https://brainquests.pages.dev/10k.html)
- [CATEGORY ART](https://brainquests.pages.dev/category-art.html)
- [PERFECT CAKE MAKER](https://eduquestsfr.pages.dev/perfect-cake-maker.html)
- [CLASH RUN](https://brainquests.pages.dev/clash-run.html)
- [BRAIN TEST IQ CHALLENGE 2](https://ieduquests.web.app/brain-test-iq-challenge-2.html)
- [JUMPING FISH RAGDOLL 3D](https://brainquests.pages.dev/jumping-fish-ragdoll-3d.html)
- [CATEGORY CAT](https://eduquestses.pages.dev/category-cat.html)
- [CATEGORY GUN238](https://eduquestses.pages.dev/category-gun238.html)
- [SUMMER MAZE](https://brainquests.pages.dev/summer-maze.html)
- [CATEGORY EXPLOIT](https://eduquestses.pages.dev/category-exploit.html)
- [MERGE BALLS SHOOTER 2048 CONNECT FRUITS](https://eduquestses.pages.dev/merge-balls-shooter-2048-connect-fruits.html)
- [NUMBER TRICKY PUZZLES](https://brainquests.pages.dev/number-tricky-puzzles.html)
- [DOMINO ONLINE MULTIPLAYER](https://eduquestspt.pages.dev/domino-online-multiplayer.html)
- [CATEGORY FIGHTING](https://eduquestses.pages.dev/category-fighting.html)
- [CATEGORY STICKMAN175](https://eduquestspt.pages.dev/category-stickman175.html)
- [CATEGORY FLASH 2](https://brainquests.pages.dev/category-flash-2.html)
- [COSMIC TETRIZ PUZZLES](https://brainquests.pages.dev/cosmic-tetriz-puzzles.html)
- [INDEX14](https://brainquests.pages.dev/index14.html)
- [FASHION PRINCESS DRESS UP](https://brainquests.pages.dev/fashion-princess-dress-up.html)
- [ONLINE PORTAL](https://brainquests.github.io/)
- [LORENZO THE RUNNER](https://brainquests.pages.dev/lorenzo-the-runner.html)
- [LITTLE LILY HALLOWEEN PREP](https://brainquests.pages.dev/little-lily-halloween-prep.html)
- [SITEMAP](https://studyquests.github.io/sitemap.html)
- [INDEX8](https://brainquests.pages.dev/index8.html)
- [STOCKINGS DILEMMA](https://brainquests.pages.dev/stockings-dilemma.html)
- [SPRUNKI PHASE BRAINROT](https://brainquests.pages.dev/sprunki-phase-brainrot.html)
- [TERMS](https://brainquests.onrender.com/terms.html)
- [CATEGORY FPS 2](https://eduquestses.pages.dev/category-fps-2.html)
- [IDLE TRADE ROUTES](https://welearnaction.onrender.com/idle-trade-routes.html)
- [WILD WEST MATCH](https://ieduquests.web.app/wild-west-match.html)
- [CATEGORY PUZZLE](https://eduquestses.pages.dev/category-puzzle.html)
- [TURBO TRUCKS RACE](https://eduquestspt.pages.dev/turbo-trucks-race.html)
- [BRICK MATCH](https://eduquestspt.pages.dev/brick-match.html)
- [CATEGORY FLASH](https://brainquests.pages.dev/category-flash.html)
- [UNICORN PRINCESS DRESS UP](https://brainquests.pages.dev/unicorn-princess-dress-up.html)
- [LOST ADVENTURE](https://eduquestspt.pages.dev/lost-adventure.html)
- [HIDDEN OBJECT STREET OF SECRETS](https://eduquestspt.pages.dev/hidden-object-street-of-secrets.html)
- [CATEGORY ESCAPE187](https://eduquestses.pages.dev/category-escape187.html)
- [SUPERMARKET SORT N MATCH](https://brainquests.pages.dev/supermarket-sort-n-match.html)
- [CATEGORY BUILDING](https://brainquests.pages.dev/category-building.html)
- [MY DOGY VIRTUAL PET](https://eduquestspt.pages.dev/my-dogy-virtual-pet.html)
- [SPRUNKI 3D SHOOTER](https://learnaction.netlify.app/sprunki-3d-shooter.html)
- [ARROW COUNT MASTER](https://ieduquests.web.app/arrow-count-master.html)
- [SUMMER ONET CONNECT](https://ieduquests.web.app/summer-onet-connect.html)
- [OBBY THE LEGENDARY DRAGON](https://eduquestses.pages.dev/obby-the-legendary-dragon.html)
- [PUSH THE COLORS](https://brainquests.pages.dev/push-the-colors.html)
- [CATEGORY BATTLESHIP](https://eduquestkr.pages.dev/category-battleship.html)
- [CATEGORY CONTROLLER 2](https://eduquestses.pages.dev/category-controller-2.html)
- [CATEGORY MAGIC46](https://eduquestses.pages.dev/category-magic46.html)
- [PET RUNNER](https://eduquestses.pages.dev/pet-runner.html)
- [CATEGORY RACING DRIVING](https://learnaction.netlify.app/category-racing-driving.html)
- [ROYAL GARDEN MATCH](https://eduquests.pages.dev/royal-garden-match.html)
- [CARD QUEST 10 MINUTE ADVENTURE](https://brainquests.pages.dev/card-quest-10-minute-adventure.html)
- [CATEGORY SOCCER60](https://learnaction.github.io/category-soccer60.html)
- [CATEGORY ART32](https://brainquests.pages.dev/category-art32.html)
- [SNAKE KING](https://eduquests.pages.dev/snake-king.html)
- [TERMS](https://brainquests.pages.dev/terms.html)
- [ITALIAN BRAINROT QUIZ](https://learnaction.netlify.app/italian-brainrot-quiz.html)
- [POOL 8](https://welearnaction.onrender.com/pool-8.html)
- [CATEGORY SNAKE40](https://eduquestses.pages.dev/category-snake40.html)
- [PIZZA MAKER COOKING GAMES FOR KIDS](https://brainquests.pages.dev/pizza-maker-cooking-games-for-kids.html)
- [COMBINATIONS DAILY](https://welearnaction.onrender.com/combinations-daily.html)
- [STACK SORTING](https://eduquestspt.pages.dev/stack-sorting.html)
- [ANOMALY CONTENT RECORD](https://eduquests.pages.dev/anomaly-content-record.html)
- [THE SPECIMEN ZERO](https://eduquestspt.pages.dev/the-specimen-zero.html)
- [MAKE IT BOOM](https://learnaction.netlify.app/make-it-boom.html)
- [WENDY SOFT GIRL MAKEUP](https://welearnaction.onrender.com/wendy-soft-girl-makeup.html)
- [WONDERS OF EGYPT MATCH](https://brainquests.pages.dev/wonders-of-egypt-match.html)
- [INDEX5](https://welearnaction.onrender.com/index5.html)
- [CATEGORY ARMY40](https://welearnaction.onrender.com/category-army40.html)
- [SWORD AND SPIN](https://brainquests.pages.dev/sword-and-spin.html)
- [CARD SOLITAIRE WORD GAME](https://learnaction.netlify.app/card-solitaire-word-game.html)
- [VALENTINES HIDDEN ALPHAWORDS](https://brainquests.pages.dev/valentines-hidden-alphawords.html)
- [COIN MERGE](https://welearnaction.onrender.com/coin-merge.html)
- [MINI OBBY WAR GAME](https://ieduquests.web.app/mini-obby-war-game.html)
- [HOUSE OF CELESTINA](https://eduquestses.pages.dev/house-of-celestina.html)
- [UNSCREW WOOD PUZZLE](https://brainquests.pages.dev/unscrew-wood-puzzle.html)
- [CATEGORY BATTLE524](https://welearnaction.onrender.com/category-battle524.html)
