import torch, numpy as np, pypose as pp
from torchvision.datasets.utils import download_url


class TestAPERPE:

    def read_tum_file(self, file_path: str):
        stamp = np.loadtxt(file_path, comments='#', delimiter=' ', usecols=(0))
        trans = np.loadtxt(file_path, comments='#', delimiter=' ', usecols=(1, 2, 3))
        rot = np.loadtxt(file_path, comments='#', delimiter=' ', usecols=(4, 5, 6, 7))

        stamp = torch.from_numpy(stamp)
        rot_q = torch.from_numpy(rot).float()
        trans = torch.from_numpy(trans).float()

        pose_torch = torch.cat((trans, rot_q), dim=-1)
        pose = pp.SE3(pose_torch)

        return stamp, pose

    def test_aperpe(self):
        local = './test/data'
        remote = 'https://raw.githubusercontent.com/MichaelGrupp/evo/master/test/data/'
        download_url(remote + '/fr2_desk_groundtruth.txt', local)
        download_url(remote + '/fr2_desk_ORB.txt', local)
        tstamp, tpose = self.read_tum_file(local + '/fr2_desk_groundtruth.txt')
        estamp, epose = self.read_tum_file(local + '/fr2_desk_ORB.txt')
        ape = pp.metric.ape(tstamp, tpose, estamp, epose, thresh=0.4, otype='SSE')
        rpe = pp.metric.rpe(tstamp, tpose, estamp, epose, thresh=0.4)
        assert isinstance(ape, torch.Tensor)
        assert isinstance(rpe, dict)


if __name__=="__main__":
    spline = TestAPERPE()
    spline.test_aperpe()
