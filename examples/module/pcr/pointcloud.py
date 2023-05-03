import torch, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torchvision.datasets.utils import download_and_extract_archive


def load_bunny(root, device=None):
    download_and_extract_archive('https://github.com/pypose/pypose/releases/'\
                                 'download/v0.4.2/bunny.pt.zip', root)
    pc1, pc2, tf = torch.load(os.path.join(root, 'bunny.pt'), map_location=device)
    return pc1, pc2, tf


def pointcloud_plot(algorithm_name, source, target, save=None, show=False):

    source = source.cpu().numpy()
    target = target.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source[..., 0], source[..., 1], source[..., 2], c='r', s=1, alpha=0.2)
    ax.scatter(target[..., 0], target[..., 1], target[..., 2], c='b', s=1, alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='source points',
                markerfacecolor='r', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='target points',
                markerfacecolor='b', markersize=5)]

    ax.legend(handles=legend_elements, loc='upper right')
    plt.title("PyPose %s" % algorithm_name)

    if save is not None:
        figure = os.path.join(save, algorithm_name + '.png')
        plt.savefig(figure)
        print("Saved to", figure)

    if show:
        plt.show()
