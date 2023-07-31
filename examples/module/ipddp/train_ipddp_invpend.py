import os
import time
import torch
import shutil
import argparse
import pypose as pp
import pickle as pkl
import torch.optim as optim
from pypose.module.ipddp import IPDDP
from tests.module.test_ipddp import InvPend

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

    state_all =      torch.zeros(n_batch, T+1,  ns, device=device)
    input_all = 0.02*torch.ones(n_batch,    T,  nc, device=device)
    state_all[...,0,:] = state

    init_traj = {'state': state_all,
                 'input': input_all}

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
        _fp_best_list = [None for batch_id in range(n_batch)]
        for batch_id in range(n_batch): # solved separated
            stage_cost = pp.module.QuadCost(Q[batch_id:batch_id+1], R[batch_id:batch_id+1], S[batch_id:batch_id+1], c[batch_id:batch_id+1])
            terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id:batch_id+1,0:1,:,:], R[batch_id:batch_id+1,0:1,:,:], S[batch_id:batch_id+1,0:1,:,:], c[batch_id:batch_id+1,0:1])
            lincon = pp.module.LinCon(gx[batch_id:batch_id+1], gu[batch_id:batch_id+1], g[batch_id:batch_id+1])
            init_traj_sample = {'state': init_traj['state'][batch_id:batch_id+1],
                                'input': init_traj['input'][batch_id:batch_id+1]}
            _ipddp_list[batch_id] = IPDDP(sys_, stage_cost, terminal_cost, lincon,
                                    gx.shape[-2], init_traj_sample)
            # detached version to solve the best traj
            with torch.no_grad():
                _fp_best_list[batch_id] = _ipddp_list[batch_id].solver()

        x_pred, u_pred, _ = _ipddp_list[0].forward(_fp_best_list) # call any one class instantiation, but perform batch grad computation
        print('x_pred solved')
        sys = InvPend(dt, length=expert['param'])
        ipddp_list = [None for batch_id in range(n_batch)]
        fp_list = [None for batch_id in range(n_batch)]
        for batch_id in range(n_batch):
            stage_cost = pp.module.QuadCost(Q[batch_id:batch_id+1], R[batch_id:batch_id+1], S[batch_id:batch_id+1], c[batch_id:batch_id+1])
            terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id:batch_id+1,0:1,:,:], R[batch_id:batch_id+1,0:1,:,:], S[batch_id:batch_id+1,0:1,:,:], c[batch_id:batch_id+1,0:1])
            lincon = pp.module.LinCon(gx[batch_id:batch_id+1], gu[batch_id:batch_id+1], g[batch_id:batch_id+1])
            init_traj_sample = {'state': init_traj['state'][batch_id:batch_id+1],
                                'input': init_traj['input'][batch_id:batch_id+1]}
            ipddp_list[batch_id] = IPDDP(sys, stage_cost, terminal_cost, lincon,
                                    gx.shape[-2], init_traj_sample)
            fp_list[batch_id] = ipddp_list[batch_id].solver()
        x_true, u_true = torch.cat([fp_list[batch_id].x for batch_id in range(n_batch)],dim=0), \
                         torch.cat([fp_list[batch_id].u for batch_id in range(n_batch)],dim=0)
        print('x_true solved')

        traj_loss = 1e4*(torch.mean((u_true - u_pred)**2) + \
                    torch.mean((x_true - x_pred)**2))
        return traj_loss

    opt = optim.SGD([param], lr=2e-1)

    for i in range(50):

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
    os.system('python .\plot.py "{}" &'.format(args.save))
if __name__=='__main__':
    main()
