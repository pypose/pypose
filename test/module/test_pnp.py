import os
import pypose as pp
import torch
import scipy
import urllib
import numpy as np

def load_data(path):        
    data = scipy.io.loadmat(path)
    points = data['point']
    
    keys = ['camPts', 'Ximg_true', 'imgPts_true', 'Ximg', 'imgPts', 'objPts']
    res = {key: list() for key in keys}

    res['camMat'] = data['A'][:, :-1].astype(int) # discard the last roll (all zeros)
    res['camMat'] = torch.from_numpy(res['camMat'])
    res['Rt'] = torch.from_numpy(data['Rt'])
    for point in points[0]:
        for idx, key in enumerate(keys):
            res[key].append(torch.from_numpy(point[idx]))
    for key in keys:
        res[key] = torch.stack(res[key])
        res[key] = res[key].squeeze(-1)
        # make a copy of the data for batch size of 2
        res[key] = res[key].unsqueeze(0)[[0, 0,]]
        
    res['camMat'] = res['camMat'].unsqueeze(0)[[0, 0,]].to(res['objPts'])
    res['Rt'] = res['Rt'].unsqueeze(0)[[0, 0,]]
    return res

class TestEPnP:
    def test_epnp_5pts(self):
        # TODO: Implement this
        return

    def test_epnp_10pts(self):
        # TODO: Implement this
        return

    def test_epnp_random(self):
        """
        """
        
        # load epfl's mat file
        test_mat_url = 'https://github.com/cvlab-epfl/EPnP/raw/master/matlab/data/input_data_noise.mat'
        tmp_path = '/tmp/pypose_test/input_data_noise.mat'
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        if not os.path.exists(tmp_path):
            urllib.request.urlretrieve(test_mat_url, tmp_path)
        assert os.path.exists(tmp_path), 'download testing mat failed'
        data = load_data(tmp_path)

        # instantiate epnp
        epnp = pp.module.EPnP()
        epnp.forward(data['objPts'], data['imgPts'], data['camMat'])
        
        return


if __name__ == "__main__":
    epnp_fixture = TestEPnP()
    epnp_fixture.test_epnp_5pts()
    epnp_fixture.test_epnp_10pts()
    epnp_fixture.test_epnp_random()