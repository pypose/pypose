import os,torch
import numpy as np
import pypose as pp
import torch.utils.data as Data
from torchvision.datasets.utils import download_url


class G2OPGO(Data.Dataset):
    '''
    This data is from the following paper:
    L. Carlone, R. Tron, K. Daniilidis, and F. Dellaert. Initialization Techniques for 3D
    SLAM: a Survey on Rotation Estimation and its Use in Pose Graph Optimization. In IEEE
    Intl. Conf. on Robotics and Automation (ICRA), pages 4597-4604, 2015.
    '''
    link = 'https://github.com/pypose/bae/releases/download/0.2.2/'
    def __init__(self, root, dataname, device='cpu', download=True):
        super().__init__()

        if download and not os.path.exists(os.path.join(root, dataname)):
            download_url(self.link + dataname, root, filename=dataname)

        def info2mat(info):
            mat = np.zeros((6,6))
            ix = 0
            for i in range(mat.shape[0]):
                mat[i,i:] = info[ix:ix+(6-i)]
                mat[i:,i] = info[ix:ix+(6-i)]
                ix += (6-i)
            return mat

        self.dtype = torch.get_default_dtype()
        filename = os.path.join(root, dataname)
        ids, nodes, edges, poses, infos = [], [], [], [], []
        with open(filename) as f:
            for line in f:
                line = line.split()
                if not line:
                    continue
                if line[0] == 'VERTEX_SE3:QUAT':
                    ids.append(int(line[1]))
                    nodes.append(pp.SE3(np.array(line[2:], dtype=np.float64)))
                elif line[0] == 'EDGE_SE3:QUAT':
                    edges.append((int(line[1]), int(line[2])))
                    poses.append(pp.SE3(np.array(line[3:10], dtype=np.float64)))
                    infos.append(torch.tensor(info2mat(np.array(line[10:], dtype=np.float64))))

        node_rows = {node_id: row for row, node_id in enumerate(ids)}
        if len(node_rows) != len(ids):
            raise ValueError("Duplicate vertex ID in pose graph")
        try:
            edges = [(node_rows[source], node_rows[target]) for source, target in edges]
        except KeyError as error:
            raise ValueError(f"Edge references missing vertex {error.args[0]}") from error

        self.ids = torch.tensor(ids, dtype=torch.int64)
        self.nodes = torch.stack(nodes).to(self.dtype).to(device)
        self.edges = torch.tensor(edges, dtype=torch.int64).to(device)
        self.poses = torch.stack(poses).to(self.dtype).to(device)
        self.infos = torch.stack(infos).to(self.dtype).to(device)
        assert self.ids.size(0) == self.nodes.size(0) \
               and self.edges.size(0) == self.poses.size(0) == self.infos.size(0)

    def init_value(self):
        return self.nodes.clone()

    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)
