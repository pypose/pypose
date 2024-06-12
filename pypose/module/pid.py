import torch
from torch import nn


class PID(nn.Module):
    r"""
    This class is the basic implementation of PID controller.

    A PID controller, standing for proportional-integral-derivative controller, operates
    as a feedback mechanism within a control loop, extensively utilized across industrial
    control systems and various applications where continuous modulation is essential.
    It functions by persistently determining an error value :math:`\mathbf{e}(t)`, which
    represents the difference between a targeted setpoint and the actual value of the
    process being controlled. To correct this error, the PID controller implements
    adjustments derived from three key components: the proportional term (P),
    the integral term (I), and the derivative term (D), each contributing to
    the overall control strategy.

    Args:
        kp (:obj:`Tensor`): proportional gain to the error
        ki (:obj:`Tensor`): integral gain to the integration of the error over the past time
        kd (:obj:`Tensor`): derivative gain to the current rate of change of the error

    The output of the PID can be described as

    .. math::
        \mathbf{u}(t) = \mathbf{K_p} \mathbf{e}(t) + \mathbf{K_i}
        \int_0^t \mathbf{e}(\tau)d \tau + \mathbf{K_d} \frac{d\mathbf{e}(t)}{dt}

    The discritized implmentation of the PID can be described as

    .. math::
        \mathbf{u}(t_k) = \mathbf{K_p} \mathbf{e}(t_k) + \mathbf{K_i} \sum_{j=0}^k
        \mathbf{e}(t_j) + \mathbf{K_d} (\mathbf{e}(t_k) - \mathbf{e}(t_{k-1}))

    """
    def __init__(self, kp, ki, kd):
        super().__init__()
        self.integrity_initialized = False
        self.integity = None
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def forward(self, error, error_dot, ff=None):
        r"""
        Args:
            error: error value :math:`\mathbf{e}(t)` as the difference between a
                desired setpoint and a measured
                value.
            error_dot: The rate of the change of error value.
            ff: feedforward system input appened to the output of the pid controller
        """
        if not self.integrity_initialized:
            self.integity = torch.zeros_like(error)
            self.integrity_initialized = True

        self.integity += error

        if ff == None:
            ff = torch.zeros_like(error)

        return self.kp * error + self.ki * self.integity + self.kd * error_dot + ff

    def reset(self):
        r"""
        This method is used to reset the internal error integrity.
        """
        if self.integrity_initialized:
            self.integity = None
            self.last_error = None
            self.integrity_initialized = False
