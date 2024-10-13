import torch
import numpy as np
import pypose as pp
from pypose.function.metric import compute_APE, StampedSE3

def csv_read_matrix(file_path:str, column:tuple[int],
                    delim: str=',', comment_str:str ="#"):
    mat = np.loadtxt(file_path,  comments=comment_str,
                     delimiter=delim, usecols=column)
    return mat

def read_tum_pose_file(file_path: str):
    tstamp = csv_read_matrix(file_path, column=(0),
                            delim=' ', comment_str='#')
    trans = csv_read_matrix(file_path, column=(1,2,3), delim=' ',
                            comment_str='#')
    rot = csv_read_matrix(file_path, column=(4,5,6,7),
                            delim=' ', comment_str='#')

    tstamp = torch.from_numpy(tstamp)
    rot_q = torch.from_numpy(rot).float()
    trans = torch.from_numpy(trans).float()

    print(f"Loaded {len(tstamp)} stamps and poses from: {file_path}")

    pose_torch = torch.cat((trans, rot_q), dim=-1)
    pose = pp.SE3(pose_torch)

    return tstamp, pose

if __name__ == '__main__':
    # The txt can be obtained from evo github
    t_gt_stamp, pose_gt = read_tum_pose_file('fr2_desk_groundtruth.txt')
    t_est_stamp, pose_est = read_tum_pose_file('fr2_desk_ORB.txt')
    traj_gt = StampedSE3(pose_gt, t_gt_stamp)
    traj_est = StampedSE3(pose_est, t_est_stamp)

    result = compute_APE(traj_gt, traj_est, match_thresh = 0.4)
    pass
