import torch, numpy as np, pypose as pp
from pypose.metric import ape, rpe
from torchvision.datasets.utils import download_url


def read_tum_pose_file(file_path: str):
    stamp = np.loadtxt(file_path, comments='#', delimiter=' ', usecols=(0))
    trans = np.loadtxt(file_path, comments='#', delimiter=' ', usecols=(1,2,3))
    rot = np.loadtxt(file_path, comments='#', delimiter=' ', usecols=(4,5,6,7))

    stamp = torch.from_numpy(stamp)
    rot_q = torch.from_numpy(rot).float()
    trans = torch.from_numpy(trans).float()

    pose_torch = torch.cat((trans, rot_q), dim=-1)
    pose = pp.SE3(pose_torch)

    return stamp, pose

if __name__ == '__main__':
    download_url('https://raw.githubusercontent.com/MichaelGrupp/evo/master/'\
                  'test/data/fr2_desk_groundtruth.txt', \
                  './test/data')
    download_url('https://raw.githubusercontent.com/MichaelGrupp/evo/master/'\
                  'test/data/fr2_desk_ORB.txt', \
                  './test/data')
    stamp_gt, pose_gt = read_tum_pose_file('./test/data/fr2_desk_groundtruth.txt')
    stamp_est, pose_est = read_tum_pose_file('./test/data/fr2_desk_ORB.txt')

    result_ape = ape(stamp_gt, pose_gt, stamp_est, pose_est, thresh = 0.4)
    result_rpe = rpe(stamp_gt, pose_gt, stamp_est, pose_est, thresh = 0.4)
