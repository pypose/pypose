import torch
import pypose as pp


class Bicycle(pp.module.System):
    '''
    This is an implementation of the 2D Bicycle kinematic model,
    see: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model
    -81cac6420357
    The robot is given a rotational and forward velocity, and traverses the 2D plane accordingly.
    This model is Nonlinear Time Invariant (NTI) and can be filtered with the ``pp.module.EKF``
    and  ``pp.module.UKF``.
    '''

    def __init__(self):
        super().__init__()

    def state_transition(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.System``.
        '''
        theta = state[..., 2] + input[1]
        vx = input[0] * theta.cos()
        vy = input[0] * theta.sin()
        return torch.stack([state[..., 0] + vx, state[..., 1] + vy, theta], dim=-1)

    def observation(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.System``.
        '''
        return state
