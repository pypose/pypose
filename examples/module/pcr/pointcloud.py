import torch, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from torchvision.datasets.utils import download_and_extract_archive


class Pointcloud():
    '''
    This is a point cloud data loader.
    '''
    def __init__():
        super().__init__()

    def load_pointcloud():
        download_and_extract_archive('https://github.com/pypose/pypose/releases/'\
                                     'download/v0.4.2/icp-test-data.pt.zip',\
                                     './examples/module/pcr/data')
        loaded_tensors = torch.load('./examples/module/pcr/data/icp-test-data.pt')
        pc1 = loaded_tensors['pc1'].squeeze(-3)
        pc2 = loaded_tensors['pc2'].squeeze(-3)
        return pc1, pc2

def pointcloud_plot(algorithm_name, new, ref, reg, elev=90, azim=-90,
                    save=None, show=False):
    new = new.cpu().numpy()
    ref = ref.cpu().numpy()
    reg = reg.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new[..., 0], new[..., 1], new[..., 2],
               c='b', s=1, alpha=0.1)
    ax.scatter(ref[..., 0], ref[..., 1], ref[..., 2],
               c='r', s=1, alpha=0.1)
    ax.scatter(reg[..., 0], reg[..., 1], reg[..., 2],
               c='k', s=1, alpha=0.1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='new points',
           markerfacecolor='b', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='reference points',
           markerfacecolor='r', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='registered points',
           markerfacecolor='k', markersize=5)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.title("%s Example" % algorithm_name.upper())

    if save is not None:
        figure = os.path.join(save, algorithm_name + ' point cloud.png')
        plt.savefig(figure)
        print("Saved to", figure)

    if show:
        plt.show()
