import torch
from torch import nn, finfo


class Constant(object):
    r'''
    Constant damping strategy used in the Levenberg-Marquardt (LM) algorithm.

    Args:
        damping (float, optional): damping factor of LM optimizer. Defaults: 1e-6.

    Example:
        >>> class PoseInv(nn.Module):
        ...     def __init__(self, *dim):
        ...         super().__init__()
        ...         self.pose = pp.Parameter(pp.randn_SE3(*dim))
        ... 
        ...     def forward(self, inputs):
        ...         return (self.pose @ inputs).Log().tensor()
        ... 
        ... device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ... inputs = pp.randn_SE3(2, 2).to(device)
        ... invnet = PoseInv(2, 2).to(device)
        ... strategy = pp.optim.strategy.Constant(damping=1e-6)
        ... optimizer = pp.optim.LM(invnet, strategy=strategy)
        ... 
        ... for idx in range(10):
        ...     loss = optimizer.step(inputs)
        ...     print('Pose loss %.7f @ %dit'%(loss, idx))
        ...     if loss < 1e-5:
        ...         print('Early Stoping!')
        ...         print('Optimization Early Done with loss:', loss.item())
        ...         break
        Pose loss 0.0000000 @ 0it
        Early Stoping!
        Optimization Early Done with loss: 9.236661990819073e-10
    
    Note:
        More details about optimization go to :meth:`pypose.optim.LevenbergMarquardt`.
    '''
    def __init__(self, damping=1e-6):
        assert damping > 0, ValueError("damping has to be positive: {}".format(damping))
        self.defaults = {'damping': damping}

    def update(self, pg, *args, **kwargs):
        pg['damping'] = pg['damping']


class Adaptive(object):
    r'''
    Adaptive damping strategy used in the Levenberg-Marquardt (LM) algorithm.

    It scales down the damping factor if the optimization step is very successful, unchanges the
    factor if the step is successful, and scales the factor up if the step is unsuccessful.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda ~\text{(damping)},
                             \bm{f}(\bm{\theta})~(\text{model}), \theta_h~(\text{high}),
                             \theta_l~(\text{low}), \delta_u~(\text{up}), \delta_d~(\text{down}) \\
            &\rule{113mm}{0.4pt}                                                                 \\
            & \rho = \frac{ \|\bm{f}(\bm{\theta})\|^2 - \|\bm{f}(\bm{\theta} + \delta)\|^2}
                      {\|\bm{f}(\bm{\theta})\|^2 - \|\bm{f}(\bm{\theta}) + \mathbf{J}\delta\|^2}
                    ~\text{(step quality)}                                                       \\
            &\textbf{if} ~~ \rho > \theta_h ~ \text{(``very successful'' step)}                  \\
            &\hspace{5mm} \lambda \leftarrow \delta_d \cdot \lambda                              \\
            &\textbf{elif} ~~ \rho > \theta_l ~ \text{(``successful'' step)}                     \\
            &\hspace{5mm} \lambda \leftarrow \lambda                                             \\
            &\textbf{else} ~ \text{(``unsuccessful'' step)}                                      \\
            &\hspace{5mm} \lambda \leftarrow \delta_u  \cdot \lambda                             \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \lambda                                                      \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        damping (float, optional): damping factor of LM optimizer. Defaults: 1e-6.
        high (float, optional): high threshold for scaling down the damping factor.
            Defaults: 0.5.
        low (float, optional): low threshold for scaling up the damping factor.
            Defaults: 1e-3.
        down (float, optional): the down scaling factor in the range of :math:`(0,1)`.
            Defaults: 0.5.
        up (float, optional): the up scaling factor in the range of :math:`(1,\infty)`.
            Defaults: 2.0.
        min (float, optional): the lower-bound of damping factor. Default: 1e-6.
        max (float, optional): the upper-bound of damping factor. Default: 1e16.
    Note:
        More details about the optimization process go to
        :meth:`pypose.optim.LevenbergMarquardt`.

    Example:
        >>> class PoseInv(nn.Module):
        ...     def __init__(self, *dim):
        ...         super().__init__()
        ...         self.pose = pp.Parameter(pp.randn_SE3(*dim))
        ... 
        ...     def forward(self, inputs):
        ...         return (self.pose @ inputs).Log().tensor()
        ... 
        ... device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ... inputs = pp.randn_SE3(2, 2).to(device)
        ... invnet = PoseInv(2, 2).to(device)
        ... strategy = pp.optim.strategy.Adaptive(damping=1e-6)
        ... optimizer = pp.optim.LM(invnet, strategy=strategy)
        ... 
        ... for idx in range(10):
        ...     loss = optimizer.step(inputs)
        ...     print('Pose loss %.7f @ %dit'%(loss, idx))
        ...     if loss < 1e-5:
        ...         print('Early Stoping!')
        ...         print('Optimization Early Done with loss:', loss.item())
        ...         break
        Pose loss 0.0000000 @ 0it
        Early Stoping!
        Optimization Early Done with loss: 9.236661990819073e-10
    '''
    def __init__(self, damping=1e-6, high=0.5, low=1e-3, up=2., down=.5, min=1e-6, max=1e16):
        assert damping > 0, ValueError("damping has to be positive: {}".format(damping))
        assert high > 0, ValueError("high has to be positive: {}".format(th1))
        assert low > 0, ValueError("low for decrease has to be positive: {}".format(th2))
        assert 0 < down < 1, ValueError("down factor has to be larger than 1: {}".format(down))
        assert 1 < up, ValueError("up factor has to be larger than 1: {}".format(up))
        self.defaults = {'damping': damping, 'high': high, 'low': low, 'up': up, 'down': down}
        self.min, self.max = min, max

    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        quality = (last - loss) / -((J @ D).mT @ (2 * R + J @ D)).squeeze()
        if quality > pg['high']:
            pg['damping'] = pg['damping'] * pg['down']
        elif quality > pg['low']:
            pg['damping'] = pg['damping']
        else:
            pg['damping'] = pg['damping'] * pg['up']
        pg['damping'] = max(self.min, min(pg['damping'], self.max))


class TrustRegion(object):
    r'''
    The trust region algorithm used in the Levenberg-Marquardt (LM) algorithm.

    This implementation follows the `trust region in Ceres <https://tinyurl.com/38r8e77x>`_.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda ~\text{(damping)},
                             \bm{f}(\bm{\theta})~(\text{model}), \theta_h~(\text{high}),
                             \theta_l~(\text{low}), \delta_u~(\text{up}), \delta_d~(\text{down}) \\
            &\rule{113mm}{0.4pt}                                                                 \\
            & \rho = \frac{ \|\bm{f}(\bm{\theta})\|^2 - \|\bm{f}(\bm{\theta} + \delta)\|^2}
                      {\|\bm{f}(\bm{\theta})\|^2 - \|\bm{f}(\bm{\theta}) + \mathbf{J}\delta\|^2}
                    ~\text{(step quality)}                                                       \\
            &\textbf{if} ~~ \rho > \theta_h ~ \text{(``very successful'' step)}                  \\
            &\hspace{5mm} \lambda \leftarrow \delta_d \cdot \lambda                              \\
            &\textbf{elif} ~~ \rho > \theta_l ~ \text{(``successful'' step)}                     \\
            &\hspace{5mm} \lambda \leftarrow \lambda                                             \\
            &\textbf{else} ~ \text{(``unsuccessful'' step)}                                      \\
            &\hspace{5mm} \lambda \leftarrow \delta_u  \cdot \lambda                             \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \lambda                                                      \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        radius (float, optional): the initial radius of the trust region (positive number).
            Default: 1e6.
        min (float, optional): the lower-bound of trust region radius. Default: 1e-6.
        max (float, optional): the upper-bound of trust region radius. Default: 1e16.
        decrease (float, optional): model cost change when expanding the trust region.
            Default: 1e-1.
        factor (float, optional): factor of expanding or shrinking the trust region.
            Default: 2.0.

    Note:
        More details about quality calculated and the optimization process go to
        :meth:`pypose.optim.LevenbergMarquardt`.

    Example:
        >>> class PoseInv(nn.Module):
        ...     def __init__(self, *dim):
        ...         super().__init__()
        ...         self.pose = pp.Parameter(pp.randn_SE3(*dim))
        ... 
        ...     def forward(self, inputs):
        ...         return (self.pose @ inputs).Log().tensor()
        ... 
        ... device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ... inputs = pp.randn_SE3(2, 2).to(device)
        ... invnet = PoseInv(2, 2).to(device)
        ... strategy = pp.optim.strategy.Adaptive(damping=1e-6)
        ... optimizer = pp.optim.LM(invnet, strategy=strategy)
        ... 
        ... for idx in range(10):
        ...     loss = optimizer.step(inputs)
        ...     print('Pose loss %.7f @ %dit'%(loss, idx))
        ...     if loss < 1e-5:
        ...         print('Early Stoping!')
        ...         print('Optimization Early Done with loss:', loss.item())
        ...         break
        Pose loss 0.0000000 @ 0it
        Early Stoping!
        Optimization Early Done with loss: 9.236661990819073e-10
    '''
    def __init__(self, radius=1e6, decrease=1e-1, factor=2., min=1e-6, max=1e16):
        super().__init__()
        assert radius > 0, ValueError("trust region radius has to be positive: {}".format(radius))
        assert decrease > 0, ValueError("min decrease has to be positive: {}".format(decrease))
        assert factor > 0, ValueError("factor for decrease has to be positive: {}".format(factor))
        self.min, self.max, damping = min, max, 1 / radius
        self.defaults = {'radius':radius, 'damping':damping, 'decrease':decrease, 'factor':factor}

    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        quality = (last - loss) / -((J @ D).mT @ (2 * R + J @ D)).squeeze()
        pg['radius'] = 1. / pg['damping']
        if quality > pg['decrease']:
            pg['radius'] = 3 * pg['radius']
            pg['factor'] = torch.tensor(2.)
        elif quality > 0:
            pg['radius'] = pg['radius']
            pg['factor'] = torch.tensor(2.)
        else:
            pg['radius'] = pg['radius'] / pg['factor']
            pg['factor'] = 2 * pg['factor']
        pg['radius'] = max(self.min, min(pg['radius'], self.max))
        pg['factor'] = max(self.min, min(pg['factor'], self.max))
        pg['damping'] = 1. / pg['radius']
