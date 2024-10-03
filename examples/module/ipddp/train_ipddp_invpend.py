import os
import time
import plot
import torch
import shutil
import argparse
import pypose as pp
import pickle as pkl
import torch.optim as optim
from pypose.module.ipddp import IPDDP
from examples.module.dynamics.invpend import InvPend

def main():
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_state', type=int, default=2)
    parser.add_argument('--n_input', type=int, default=1)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = '.'.join(["{}={}".format(x, getattr(args, x))
                  for x in ['n_state', 'n_input', 'T']])
    if args.save is None:
        args.save = os.path.join(args.work, 'invpend', t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = 'cuda' if args.cuda else 'cpu'

    ns, nc = args.n_state, args.n_input
    T = args.T
    expert_seed = 26
    state = torch.tensor([[-2.,0.],
                          [-1., 0.]],
                          device=device)
    n_batch = state.shape[0]
    input_all = 0.02*torch.ones(n_batch,    T,  nc, device=device)

    assert expert_seed != args.seed
    torch.manual_seed(expert_seed)

    dt = 0.05   # Delta t
    Q = torch.tile(dt*torch.eye(ns, ns, device=device), (n_batch, T, 1, 1))
    R = torch.tile(dt*torch.eye(nc, nc, device=device), (n_batch, T, 1, 1))
    S = torch.tile(torch.zeros(ns, nc, device=device), (n_batch, T, 1, 1))
    c = torch.tile(torch.zeros(1, device=device), (n_batch, T))

    # Create constraint object
    gx = torch.tile(torch.zeros(2*nc, ns, device=device), (n_batch, T, 1, 1))
    gu = torch.tile(torch.vstack((torch.eye(nc, nc, device=device), - torch.eye(nc, nc, device=device)) ), (n_batch, T, 1, 1))
    g = torch.tile(torch.hstack((-0.25 * torch.ones(nc, device=device), -0.25 * torch.ones(nc, device=device)) ), (n_batch, T, 1))
    batch_lincon = pp.module.LinCon(gx, gu, g)


    expert = dict(
        param = torch.Tensor([10.0]).to(device)
    )
    fname = os.path.join(args.save, 'expert.pkl')
    with open(fname, 'wb') as fi:
        pkl.dump(expert, fi)

    torch.manual_seed(args.seed)
    param = torch.Tensor([8.0]).to(device).requires_grad_()

    fname = os.path.join(args.save, 'pypose losses.csv')
    loss_f = open(fname, 'w')
    loss_f.write('im_loss,mse\n')
    loss_f.flush()

    fname = os.path.join(args.save, 'pypose time.csv')
    time_d = open(fname, 'w')
    time_d.write('backward_time,overall_time\n')
    time_d.flush()

    def get_loss(_param):
        sys_ = InvPend(dt, length=_param)
        _ipddp_list = [None for batch_id in range(n_batch)]
        x_pred_list = [None for batch_id in range(n_batch)]
        u_pred_list = [None for batch_id in range(n_batch)]
        for batch_id in range(n_batch): # solved separated
            stage_cost = pp.module.QuadCost(Q[batch_id:batch_id+1], R[batch_id:batch_id+1], S[batch_id:batch_id+1], c[batch_id:batch_id+1])
            terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id:batch_id+1,0:1,:,:], R[batch_id:batch_id+1,0:1,:,:], S[batch_id:batch_id+1,0:1,:,:], c[batch_id:batch_id+1,0:1])
            lincon = pp.module.LinCon(gx[batch_id:batch_id+1], gu[batch_id:batch_id+1], g[batch_id:batch_id+1])
            _ipddp_list[batch_id] = IPDDP(sys_, stage_cost, terminal_cost, lincon, T, B=(1,))
            # solve the best traj with computational graph
            x_init, u_init = state[batch_id:batch_id+1], input_all[batch_id:batch_id+1]
            x_pred_list[batch_id], u_pred_list[batch_id], _= _ipddp_list[batch_id].forward(x_init, u_init)

        x_pred, u_pred=  torch.cat(x_pred_list,dim=0), torch.cat(u_pred_list,dim=0)
        print('x_pred solved')
        sys = InvPend(dt, length=expert['param'])
        ipddp_list = [None for batch_id in range(n_batch)]
        x_true_list = [None for batch_id in range(n_batch)]
        u_true_list = [None for batch_id in range(n_batch)]
        for batch_id in range(n_batch):
            stage_cost = pp.module.QuadCost(Q[batch_id:batch_id+1], R[batch_id:batch_id+1], S[batch_id:batch_id+1], c[batch_id:batch_id+1])
            terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id:batch_id+1,0:1,:,:], R[batch_id:batch_id+1,0:1,:,:], S[batch_id:batch_id+1,0:1,:,:], c[batch_id:batch_id+1,0:1])
            lincon = pp.module.LinCon(gx[batch_id:batch_id+1], gu[batch_id:batch_id+1], g[batch_id:batch_id+1])
            ipddp_list[batch_id] = IPDDP(sys, stage_cost, terminal_cost, lincon, T, B=(1,))
            x_init, u_init = state[batch_id:batch_id+1], input_all[batch_id:batch_id+1]
            with torch.no_grad():
                x_true_list[batch_id], u_true_list[batch_id], _ = ipddp_list[batch_id].forward(x_init, u_init)
        x_true, u_true=  torch.cat(x_true_list,dim=0), torch.cat(u_true_list,dim=0)
        print('x_true solved')

        traj_loss = 1e4*(torch.mean((u_true - u_pred)**2) + \
                    torch.mean((x_true - x_pred)**2))
        return traj_loss

    opt = optim.SGD([param], lr=2e-1)

    for i in range(4):

        t1 = time.time()
        traj_loss = get_loss(param)
        opt.zero_grad()
        t2 = time.time()
        with torch.autograd.set_detect_anomaly(True):
            traj_loss.backward()
            print('grad', param.grad)
        t3 = time.time()
        backward_time = t3 - t2
        opt.step()
        t4 = time.time()
        overall_time = t4 - t1

        model_loss = torch.mean((param - expert['param'])**2)

        loss_f.write('{},{}\n'.format(traj_loss.item(), model_loss.item()))
        loss_f.flush()

        time_d.write('{},{}\n'.format(backward_time, overall_time))
        time_d.flush()

        plot_interval = 1
        if i % plot_interval == 0:
            print(param, expert['param'])
        print('{:04d}: traj_loss: {:.4f} model_loss: {:.4f}'.format(
            i, traj_loss.item(), model_loss.item()))

    print( args.save)
    plot.plot(args.save)
if __name__=='__main__':
    main()
