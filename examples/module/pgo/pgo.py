import os
import torch
import warnings
import argparse
import numpy as np
import pypose as pp
from torch import nn
from pgo_dataset import G2OPGO
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

 
class PoseGraph(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        error = node1.Inv() @ node2 @ poses.Inv()
        return error.Log().tensor()


@torch.no_grad()
def plot_and_save(points, pngname, title='', axlim=None):
    points = points.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:,0], points[:,1], points[:,2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    plt.savefig(pngname)
    print('Saving to', pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Graph Optimization')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--radius", type=float, default=1e4, help="trust region radius")
    parser.add_argument("--save", type=str, default='./examples/module/pgo/save/', help="location of png files to save")
    parser.add_argument("--dataroot", type=str, default='./examples/module/pgo/pgodata', help="dataset location downloaded")
    parser.add_argument("--dataname", type=str, default='parking-garage.g2o', help="dataset name")
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    data = G2OPGO(args.dataroot, args.dataname, device=args.device)
    edges, poses, infos = data.edges, data.poses, data.infos

    graph = PoseGraph(data.nodes).to(args.device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=args.radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=False)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

    pngname = os.path.join(args.save, args.dataname+'.png')
    axlim = plot_and_save(graph.nodes.translation(), pngname, args.dataname)

    ### the 1st implementation: for customization and easy to extend
    while scheduler.continual:
        loss = optimizer.step(input=(edges, poses), weight=infos)
        scheduler.step(loss)

        name = os.path.join(args.save, args.dataname + '_' + str(scheduler.steps))
        title = 'PyPose PGO at the %d step(s) with loss %7f'%(scheduler.steps, loss.item())
        plot_and_save(graph.nodes.translation(), name+'.png', title, axlim=axlim)
        torch.save(graph.state_dict(), name+'.pt')

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    scheduler.optimize(input=(edges, poses), weight=infos)
