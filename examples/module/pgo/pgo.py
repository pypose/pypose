import os
import torch
import argparse
import pypose as pp
from torch import nn
from pgo_dataset import G2OPGO
import matplotlib.pyplot as plt
import pypose.optim.solver as ppos
import pypose.optim.strategy as ppost
from pypose.autograd.function import psjac
from pypose.optim.scheduler import StopOnPlateau


class PoseGraph(nn.Module):

    def __init__(self, nodes, sparse=True):
        super().__init__()
        self.nodes = pp.Parameter(nodes, sjac=sparse)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        return PoseGraph.error(node1, node2, poses)

    @psjac
    def error(node1, node2, poses):
        error = poses.Inv() @ node1.Inv() @ node2
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
    parser.add_argument("--device", type=str, default='cuda', help="cuda or cpu")
    parser.add_argument("--radius", type=float, default=1e4, help="trust region radius")
    parser.add_argument("--save", type=str, default='./examples/module/pgo/save/', \
                        help="files location to save")
    parser.add_argument("--dataroot", type=str, default='./examples/module/pgo/data', \
                        help="dataset location")
    parser.add_argument("--dataname", type=str, default='parking-garage.g2o', \
                        help="dataset name")
    parser.add_argument('--no-sparse', dest='sparse', action='store_false', \
                        help='use dense optimization with information matrices')
    parser.add_argument('--sparse', action='store_true', \
                        help='use sparse Jacobians (information matrices are unsupported)')
    parser.add_argument('--no-vectorize', dest='vectorize', action='store_false', \
                        help='disable vectorization in dense mode to save memory (incompatible with --sparse)')
    parser.add_argument('--vectorize', action='store_true', \
                        help='vectorize dense Jacobian computation')
    parser.set_defaults(vectorize=True, sparse=True)
    args = parser.parse_args()
    assert not (args.sparse and not args.vectorize), \
        "--no-vectorize cannot be used with --sparse; use --no-sparse for dense mode"
    print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    data = G2OPGO(args.dataroot, args.dataname, device=args.device, download=True)
    edges, poses, infos = data.edges, data.poses, data.infos

    graph = PoseGraph(data.nodes, sparse=args.sparse).to(args.device)
    solver = ppos.PCG() if args.sparse else ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=args.radius)
    if args.sparse:
        optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy,
                                min=1e-6, sparse=True)
    else:
        optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy,
                                min=1e-6, vectorize=args.vectorize)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)
    weight = None if args.sparse else infos

    pngname = os.path.join(args.save, args.dataname+'.png')
    axlim = plot_and_save(graph.nodes.translation(), pngname, args.dataname)
    ### the 1st implementation: for customization and easy to extend
    while scheduler.continual():
        loss = optimizer.step(input=(edges, poses), weight=weight)
        scheduler.step(loss)

        name = os.path.join(args.save, args.dataname + '_' + str(scheduler.steps))
        title = 'PyPose PGO at the %d step(s) with loss %7f'%(scheduler.steps, loss.item())
        plot_and_save(graph.nodes.translation(), name+'.png', title, axlim=axlim)
        torch.save(graph.state_dict(), name+'.pt')

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    scheduler.optimize(input=(edges, poses), weight=weight)
