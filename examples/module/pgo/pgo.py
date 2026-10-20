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

    def __init__(self, nodes):
        super().__init__()
        self.register_buffer('node_fixed', nodes[:1].clone())
        self.nodes = pp.Parameter(nodes[1:].clone(), sjac=True)

    def nodes_all(self):
        return torch.cat((self.node_fixed, self.nodes), dim=0)

    def forward(self, edges, poses, sqrt_info):
        nodes = self.nodes_all()
        node1 = nodes[edges[..., 0]]
        node2 = nodes[edges[..., 1]]
        return PoseGraph.error(node1, node2, poses, sqrt_info)

    @psjac
    def error(node1, node2, poses, sqrt_info):
        estimate = node1.Inv() @ node2
        translation = estimate.translation() - poses.translation()
        rotation = poses.rotation() @ estimate.rotation().Inv()
        # Compute 2 * vec(q) through Log so backward uses PyPose's tangent convention.
        phi = rotation.Log().tensor()
        rotation_error = phi * torch.sinc(phi.norm(dim=-1, keepdim=True) / (2 * torch.pi))
        # Restore the quaternion's hemisphere after taking the principal Log.
        rotation_error = torch.where(rotation.tensor()[..., 3:] < 0, -rotation_error, rotation_error)
        residual = torch.cat((translation, rotation_error), dim=-1)
        return (sqrt_info @ residual.unsqueeze(-1)).squeeze(-1)

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
    plt.close()
    print('Saving to', pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Graph Optimization')
    parser.add_argument("--device", type=str, default='cuda', help="cuda device")
    parser.add_argument("--radius", type=float, default=1e2, help="initial trust region radius")
    parser.add_argument("--save", type=str, default='./examples/module/pgo/save/', \
                        help="files location to save")
    parser.add_argument("--dataroot", type=str, default='./examples/module/pgo/data', \
                        help="dataset location")
    parser.add_argument("--dataname", type=str, default='parking-garage.g2o', \
                        help="dataset name")
    parser.add_argument("--steps", type=int, default=80, help="maximum optimization steps")
    args = parser.parse_args()
    print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    data = G2OPGO(args.dataroot, args.dataname, device=args.device, download=True)
    edges, poses, infos = data.edges, data.poses, data.infos
    # Whiten inside the residual because sparse LM does not accept a weight matrix.
    sqrt_info = torch.linalg.cholesky(infos).mT

    graph = PoseGraph(data.nodes).to(args.device)
    solver = ppos.PCG()
    strategy = ppost.TrustRegion(radius=args.radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-6, sparse=True)
    scheduler = StopOnPlateau(optimizer, steps=args.steps, patience=3, decreasing=1e-3, verbose=True)

    pngname = os.path.join(args.save, args.dataname+'.png')
    axlim = plot_and_save(graph.nodes_all().translation(), pngname, args.dataname)
    ### the 1st implementation: for customization and easy to extend
    while scheduler.continual():
        loss = optimizer.step(input=(edges, poses, sqrt_info))
        scheduler.step(loss)

        name = os.path.join(args.save, args.dataname + '_' + str(scheduler.steps))
        title = 'PyPose PGO at the %d step(s) with loss %7f'%(scheduler.steps, loss.item())
        plot_and_save(graph.nodes_all().translation(), name+'.png', title, axlim=axlim)
        torch.save(graph.state_dict(), name+'.pt')

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    scheduler.optimize(input=(edges, poses, sqrt_info))
