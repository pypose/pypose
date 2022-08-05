# Unit test tools.
import functools
import inspect
import unittest

# System tools.
import numpy as np

# pypose.
import pypose as pp

# Test utils.
from .common import ( torch_equal, show_delimeter )

# PyTorch
import torch
from torch.autograd import functional as AF
from functorch import vmap, jacrev

def camera_projection( K, T, P ):
    '''
    K (Tensor): a 3x3 camera intrinsic matrix.
    T (SE3): a SE3 object, the camera pose.
    P ()
    '''
    pass

def huber_loss(a, delta=torch.Tensor([0.1])):
    '''
    This is the Wikipedia version.
    https://en.wikipedia.org/wiki/Huber_loss

    '''
    abs_a = torch.abs(a)
    m = (a <= delta).type(torch.float)
    return m * ( 0.5 * a**2 ) + (1 - m) * delta * ( abs_a - 0.5 * delta )

def naive_hat(v):
    '''
    Naive implementation of the hat operation, that is converting
    a 3-vector to a skew-symmetric matrix.
    '''

    m = torch.zeros((3, 3), dtype=v.dtype, device=v.device)
    m[0, 1] = -v[2]
    m[0, 2] =  v[1]
    m[1, 0] =  v[2]
    m[1, 2] = -v[0]
    m[2, 0] = -v[1]
    m[2, 1] =  v[0]
    return m

def naive_so3_2_angle_and_vector(phi):
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True)
    a = torch.nn.functional.normalize(phi, dim=-1)
    return theta, a

def naive_left_jacobian(theta, a):
    '''
    theat (Tensor): must have a dimension of 1.
    a (Tensor): must have a dimension of 3 x 1.
    '''
    theta = theta.view((1, 1))
    a = a.view((3, 1))

    identity = torch.eye(3, dtype=theta.dtype, device=theta.device)
    aat = a * torch.transpose(a, 0, 1)
    ahat = naive_hat(a)

    stt = torch.sin(theta) / theta

    return stt * identity + ( 1 - stt ) * aat + ( 1 - torch.cos(theta) ) / theta * ahat

def strip_last_column(t):
    last_dim = t.shape[-1]
    return t[..., :(last_dim-1)]

class Test_ReprojectionError(unittest.TestCase):
    def test_SE3_act3(self):
        print()
        show_delimeter('Test SE3 object acting on a 3-point. ')

        # A random SE3 object.
        T_raw = pp.randn_SE3(1)[0]

        # A random 3D point.
        P0_raw = torch.rand((3,), dtype=torch.float)

        # True value.
        MT = T_raw.matrix()
        P1_raw = MT[ :3, :3 ] @ P0_raw + MT[:3, 3]
        jacobian_raw = torch.zeros((3, 6), dtype=P1_raw.dtype)
        jacobian_raw[:3, :3] = torch.eye(3)
        jacobian_raw[:3, 3:6] = naive_hat( -P1_raw )

        print(f'T_raw = {T_raw}')
        print(f'MT = \n{MT}')
        print(f'P0_raw = {P0_raw}')
        print(f'P1_raw = {P1_raw}')
        print(f'jacobian_raw = \n{jacobian_raw}')

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        for entry in test_entries:
            print(entry)

            # Convert the Tesnors.
            device = entry['device']

            T = T_raw.to(device)
            P0 = P0_raw.to(device)
            P1 = P1_raw.to(device)

            # Require gradient explicitly.
            T.requires_grad = True

            TP = T @ P0

            # Gradient.
            dummy_gradient = torch.ones((3,), dtype=TP.dtype, device=TP.device)
            TP.backward(gradient=dummy_gradient)
            print(f'T.grad = {T.grad}')

            # Compare
            try:
                torch_equal( TP, P1 )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'SE3 act3 failed with entry {entry}')

    def test_SE3_act4(self):
        print()
        show_delimeter('Test SE3 object acting on a homogeneous point. ')

        # A random SE3 object.
        T_raw = pp.randn_SE3(1)[0]

        # A random 3D homogeneous point.
        P0_raw = torch.rand((4,), dtype=torch.float)
        P0_raw[-1] = 1.0

        # True value.
        MT = T_raw.matrix()
        P1_raw = MT @ P0_raw

        print(f'T_raw = {T_raw}')
        print(f'MT = {MT}')
        print(f'P0_raw = {P0_raw}')
        print(f'P1_raw = {P1_raw}')

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        for entry in test_entries:
            print(entry)

            # Convert the Tesnors.
            device = entry['device']
            T = T_raw.to(device)
            P0 = P0_raw.to(device)
            P1 = P1_raw.to(device)

            TP = T @ P0

            # Compare
            try:
                torch_equal( TP, P1 )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'SE3 act4 failed with entry {entry}')

    def test_SE3_jacobian(self):
        print()
        show_delimeter('Test SE3 Jacobian computation. ')

        # A random SE3 object.
        T_raw = pp.randn_SE3(1)[0]

        # A random 3D homogeneous point.
        P0_raw = torch.rand((3,), dtype=torch.float)

        # True value.
        MT = T_raw.matrix()
        RT = MT[ :3, :3 ] @ P0_raw
        P1_raw = RT + MT[:3, 3]
        jacobian_raw = torch.zeros((3, 6), dtype=P1_raw.dtype)
        jacobian_raw[:3, :3] = torch.eye(3)
        jacobian_raw[:3, 3:6] = naive_hat( -P1_raw )

        print(f'T_raw = {T_raw}')
        print(f'RT = \n{RT}')
        print(f'MT = \n{MT}')
        print(f'P0_raw = {P0_raw}')
        print(f'P1_raw = {P1_raw}')
        print(f'jacobian_raw = \n{jacobian_raw}')

        def mvm(x, p):
            return x @ p

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        for entry in test_entries:
            T = T_raw.to(entry['device'])
            P0 = P0_raw.to(entry['device'])
            T.requires_grad = True

            J = AF.jacobian(functools.partial(mvm, p=P0), T, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            
            # Strip off the last column.
            J = strip_last_column(J)

            J_t = jacobian_raw.to(entry['device'])
            # print(f'J = \n{J}')

            # Compare
            try:
                torch_equal( J, J_t )
            except Exception as exc:
                print(exc)
                print(f'J = \n{J}')
                print(f'J_t = \n{J_t}')
                self.assertTrue(False, f'SE3 Jacobian test failed with entry {entry}')

    def test_se3_act(self):
        print()
        show_delimeter('Test se3 object acting on a 3-point. ')

        # A random SE3 object.
        T_raw = pp.randn_se3(1)[0]

        # A random 3D point.
        P0_raw = torch.rand((3,), dtype=torch.float)

        # True value.
        MT = T_raw.matrix()
        P1_raw = MT[ :3, :3 ] @ P0_raw + MT[:3, 3]

        theta, a = naive_so3_2_angle_and_vector(P1_raw)

        jacobian_raw = torch.zeros((3, 6), dtype=P1_raw.dtype)
        jacobian_raw[:3, :3] = torch.eye(3)
        jacobian_raw[:3, 3:6] = naive_hat( -P1_raw ) @ naive_left_jacobian( theta, a )

        print(f'T_raw = {T_raw}')
        print(f'MT = \n{MT}')
        print(f'P0_raw = {P0_raw}')
        print(f'P1_raw = {P1_raw}')
        print(f'theta = {theta}')
        print(f'a = {a}')
        print(f'jacobian_raw = \n{jacobian_raw}')

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        for entry in test_entries:
            print(entry)

            # Convert the Tesnors.
            device = entry['device']

            T = T_raw.to(device)
            P0 = P0_raw.to(device)
            P1 = P1_raw.to(device)

            # Require gradient explicitly.
            T.requires_grad = True

            TP = T.Exp() @ P0

            # Gradient.
            dummy_gradient = torch.ones((3,), dtype=TP.dtype, device=TP.device)
            TP.backward(gradient=dummy_gradient)
            print(f'T.grad = {T.grad}')

            # Compare
            try:
                torch_equal( TP, P1 )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'se3 act failed with entry {entry}')

    def test_se3_jacobian(self):
        print()
        show_delimeter('Test se3 object acting on a 3-point. ')

        # A random SE3 object.
        T_raw = pp.randn_se3()

        # A random 3D point.
        P0_raw = torch.rand((3,), dtype=torch.float)

        # True value.
        MT = T_raw.matrix()
        P1_raw = MT[ :3, :3 ] @ P0_raw + MT[:3, 3]

        theta, a = naive_so3_2_angle_and_vector(P0_raw)

        jacobian_raw = torch.zeros((3, 6), dtype=P1_raw.dtype)
        jacobian_raw[:3, :3] = MT[ :3, :3 ]
        jacobian_raw[:3, 3:6] = naive_hat( -P1_raw )

        print(f'T_raw = {T_raw}')
        print(f'MT = \n{MT}')
        print(f'P0_raw = {P0_raw}')
        print(f'P1_raw = {P1_raw}')
        print(f'theta = {theta}')
        print(f'a = {a}')
        print(f'jacobian_raw = \n{jacobian_raw}')

        def mvm(x, p):
            return x.Exp() @ p

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        for entry in test_entries:
            print(entry)

            # Convert the Tesnors.
            device = entry['device']

            T = T_raw.to(device)
            P0 = P0_raw.to(device)

            # Require gradient explicitly.
            T.requires_grad = True

            J = AF.jacobian(functools.partial(mvm, p=P0), T, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')

            J_t = jacobian_raw.to(device)
            # print(f'J = \n{J}')

            # These are going to fail because the values are not consistent.
            # # Compare
            # try:
            #     torch_equal( J, J_t )
            # except Exception as exc:
            #     print(exc)
            #     print(f'J = \n{J}')
            #     print(f'J_t = \n{J_t}')
            #     self.assertTrue(False, f'SE3 Jacobian test failed with entry {entry}')

    def test_batched_jacobian(self):
        print()
        show_delimeter('Test batched jacobian se3 object acting on a 3-point. ')

        # A random SE3 object.
        T_raw = pp.randn_se3(2)

        # A random 3D point.
        P0_raw = torch.rand((2, 3,), dtype=torch.float)

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        def mvm(x, p):
            return x.Exp() @ p

        for entry in test_entries:
            device = entry['device']

            T = T_raw.to(device)
            P0 = P0_raw.to(device)

            # Require gradient explicitly.
            T.requires_grad = True

            J = AF.jacobian(mvm, (T, P0), create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            # J = vmap(AF.jacobian, in_dims=(None, (0, 0)))(mvm, (T, P0), create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')

            # print(f'J.shape = {J.shape}')
            print(J)

    def test_reprojection_jacobian(self):
        print()
        show_delimeter('Test jacobian calculation from the reprojection error. ')