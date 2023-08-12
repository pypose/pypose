#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from tqdm import tqdm

try: import setGPU
except ImportError: pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import numpy.random as npr

import sys
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import setproctitle

import models

import datetime
import math


import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

def print_header(msg):
    print('===>', msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--boardSz', type=int, default=2)
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--testBatchSz', type=int, default=2)
    parser.add_argument('--nEpoch', type=int, default=20)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--qp-solver', type=str, default='qpth', choices=['qpth', 'osqpth'])

    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True

    fcP = subparsers.add_parser('fc')
    fcP.add_argument('--nHidden', type=int, nargs='+', default=[100,100])
    fcP.add_argument('--bn', action='store_true')

    convP = subparsers.add_parser('conv')
    convP.add_argument('--nHidden', type=int, default=50)
    convP.add_argument('--bn', action='store_true')

    spOptnetEqP = subparsers.add_parser('spOptnetEq')
    spOptnetEqP.add_argument('--Qpenalty', type=float, default=0.1)

    optnetEqP = subparsers.add_parser('optnetEq')
    optnetEqP.add_argument('--Qpenalty', type=float, default=0.1)

    optnetIneqP = subparsers.add_parser('optnetIneq')
    optnetIneqP.add_argument('--Qpenalty', type=float, default=0.1)
    optnetIneqP.add_argument('--nineq', type=int, default=100)
    optnetLatent = subparsers.add_parser('optnetLatent')

    optnetLatent.add_argument('--Qpenalty', type=float, default=0.1)
    optnetLatent.add_argument('--nLatent', type=int, default=100)
    optnetLatent.add_argument('--nineq', type=int, default=100)
    args = parser.parse_args()


    args.cuda = not args.no_cuda and torch.cuda.is_available()

    t = '{}.{}'.format(args.boardSz, args.model)

    if args.model == 'optnetEq' or args.model == 'spOptnetEq':
        t += '.Qpenalty={}'.format(args.Qpenalty)
    elif args.model == 'optnetIneq':
        t += '.Qpenalty={}'.format(args.Qpenalty)
        t += '.nineq={}'.format(args.nineq)
    elif args.model == 'optnetLatent':
        t += '.Qpenalty={}'.format(args.Qpenalty)
        t += '.nLatent={}'.format(args.nLatent)
        t += '.nineq={}'.format(args.nineq)
    elif args.model == 'fc':
        t += '.nHidden:{}'.format(','.join([str(x) for x in args.nHidden]))
        if args.bn:
            t += '.bn'
    if args.save is None:
        args.save = os.path.join(args.work, t)
    setproctitle.setproctitle('bamos.sudoku.' + t)

    # number of train data
    nTrain = int(40)

    # the target robot positions and control inputs:
    # the first three elements are the target positions at t, t+1, t+2
    # the last three elements are the target control inputs at t, t+1, t+2
    # the target robot positions and control inputs can be redesigned real-tilme
    target_point = np.array([[1,1,1,0,0,0],[1,1,1,0,0,0]])

    b1 = target_point.repeat([nTrain-1,1],axis=0)

    trainX = torch.tensor(b1)

    trainY  = torch.tensor(b1)

    # number of test data 
    nTest  = int(20)

    b2 = target_point.repeat([nTest-1,1],axis=0)

    testX = torch.tensor(b2)

    testY = torch.tensor(b2)

    assert(nTrain % args.batchSz == 0)
    assert(nTest % args.testBatchSz == 0)

    # save = work/2.optnetEq.Qpenalty=0.1
    save = args.save
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    npr.seed(1)


    print_header('Building model')
    if args.model == 'fc':
        nHidden = args.nHidden
        model = models.FC(nFeatures, nHidden, args.bn)
    elif args.model == 'conv':
        model = models.Conv(args.boardSz)
    elif args.model == 'optnetEq':
        model = models.OptNetEq(
            n=args.boardSz, Qpenalty=args.Qpenalty, qp_solver=args.qp_solver,
            trueInit=False)
    elif args.model == 'spOptnetEq':
        model = models.SpOptNetEq(args.boardSz, args.Qpenalty, trueInit=False)
    elif args.model == 'optnetIneq':
        model = models.OptNetIneq(args.boardSz, args.Qpenalty, args.nineq)
    elif args.model == 'optnetLatent':
        model = models.OptNetLatent(args.boardSz, args.Qpenalty, args.nLatent, args.nineq)
    else:
        assert False

    if args.cuda:
        model = model.cuda()

    fields = ['epoch', 'loss', 'err']
    trainF = open(os.path.join(save, 'train.csv'), 'w')
    trainW = csv.writer(trainF)
    trainW.writerow(fields)
    trainF.flush()
    fields = ['epoch', 'loss', 'err']
    testF = open(os.path.join(save, 'test.csv'), 'w')
    testW = csv.writer(testF)
    testW.writerow(fields)
    testF.flush()


    if 'optnet' in args.model:
        # if args.tvInit: lr = 1e-4
        # elif args.learnD: lr = 1e-2
        # else: lr = 1e-3
        lr = 1e-1
    else:
        lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # writeParams(args, model, 'init')
    for epoch in range(1, args.nEpoch+1):
        # update_lr(optimizer, epoch)
        train(args, epoch, model, trainF, trainW, trainX, trainY, optimizer)

        test(args, epoch, model, testF, testW, testX, testY)

        torch.save(model, os.path.join(args.save, 'latest.pth'))

        os.system('./plot.py "{}" &'.format(args.save))

def writeParams(args, model, tag):
    if args.model == 'optnet':
        A = model.A.data.cpu().numpy()
        np.savetxt(os.path.join(args.save, 'A.{}'.format(tag)), A)

# @profile
def train(args, epoch, model, trainF, trainW, trainX, trainY, optimizer):

    batchSz = args.batchSz

    batch_data_t = torch.FloatTensor(batchSz, trainX.size(1))

    batch_targets_t = torch.FloatTensor(batchSz, trainY.size(1))

    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()

    batch_data = Variable(batch_data_t, requires_grad=False)

    batch_targets = Variable(batch_targets_t, requires_grad=False)

    # two random initial positions as the batchsize is set as 2
    inix1 = 1-2*torch.rand(1).double()
    inix2 = 1-2*torch.rand(1).double()

    # record initial time instant: unit(second)
    t0 = datetime.datetime.now()
    t0 =  60*t0.minute + t0.second + 0.000001*t0.microsecond
   

    for i in range(0, trainX.size(0), batchSz):

        start = time.time()

        inix1 = torch.tensor([inix1], dtype=torch.double)
        inix2 = torch.tensor([inix2], dtype=torch.double)

        batch_data.data[:] = trainX[i:i+batchSz]

        batch_targets.data[:] = trainY[i:i+batchSz]

        optimizer.zero_grad()

        # record time instant
        t1 = datetime.datetime.now()
        t1 =  60*t1.minute + t1.second + 0.000001*t1.microsecond
        t = t1-t0

        # redesign target positions at t, t+1, t+2, where '1' is time interval such as delta t
        batch_data.data[0][0] = math.sin(t)
        batch_data.data[0][1] = math.sin(t+0.06)
        batch_data.data[0][2] = math.sin(t+0.12)
        batch_data.data[0][3] = inix1

        batch_data.data[1][0] = math.cos(t)
        batch_data.data[1][1] = math.cos(t+0.06)
        batch_data.data[1][2] = math.cos(t+0.12)
        batch_data.data[1][3] = inix2

        preds = model(batch_data)

        # measurment noise
        c1 = -0.1 + 0.2*torch.rand(1).double()
        c2 = -0.1 + 0.2*torch.rand(1).double()
        c3 = -0.1 + 0.2*torch.rand(1).double()
    
        # true robot model with noises
 
        batch_targets.data[0][0] = inix1 + preds[0][3] +c1
        batch_targets.data[0][1] = batch_targets.data[0][0] + preds[0][4] +c2
        batch_targets.data[0][2] = batch_targets.data[0][1] + preds[0][5] +c3

        batch_targets.data[1][0] = inix2 + preds[1][3] +c1
        batch_targets.data[1][1] = batch_targets.data[1][0] + preds[1][4] +c2
        batch_targets.data[1][2] = batch_targets.data[1][1] + preds[1][5] +c3

        inix1 = batch_targets.data[0][0]
        inix2 = batch_targets.data[1][0]

        print(inix1)
        print(inix2)

        loss = nn.MSELoss()(preds, batch_targets)

        loss.backward()


        optimizer.step()

        err = computeErr(preds.data)/batchSz

        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} Err: {:.4f} Time: {:.2f}s'.format(
            epoch, i+batchSz, trainX.size(0),
            float(i+batchSz)/trainX.size(0)*100,
            loss.item(), err, time.time()-start))

        trainW.writerow(
            (epoch-1+float(i+batchSz)/trainX.size(0), loss.item(), err))
        trainF.flush()

def test(args, epoch, model, testF, testW, testX, testY):
    batchSz = args.testBatchSz

    test_loss = 0
    batch_data_t = torch.FloatTensor(batchSz, testX.size(1))
    batch_targets_t = torch.FloatTensor(batchSz, testY.size(1))
    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()

    with torch.no_grad():   
        batch_data = Variable(batch_data_t)
        batch_targets = Variable(batch_targets_t)

    inix1 = 1-2*torch.rand(1).double()
    inix2 = 1-2*torch.rand(1).double()

    t0 = datetime.datetime.now()
    t0 =  60*t0.minute + t0.second + 0.000001*t0.microsecond


    nErr = 0
    for i in range(0, testX.size(0), batchSz):
        print('Testing model: {}/{}'.format(i, testX.size(0)), end='\r')

        batch_data.data[:] = testX[i:i+batchSz]

        # time record

        t1 = datetime.datetime.now()
        t1 =  60*t1.minute + t1.second + 0.000001*t1.microsecond
        t = t1-t0

        batch_data.data[0][0] = math.sin(t)
        batch_data.data[0][1] = math.sin(t+0.06)
        batch_data.data[0][2] = math.sin(t+0.12)       

        batch_data.data[1][0] = math.cos(t)
        batch_data.data[1][1] = math.cos(t+0.06)
        batch_data.data[1][2] = math.cos(t+0.12)


        batch_targets.data[:] = testY[i:i+batchSz]
        output = model(batch_data)

        c1 = -0.1 + 0.2*torch.rand(1).double()
        c2 = -0.1 + 0.2*torch.rand(1).double()
        c3 = -0.1 + 0.2*torch.rand(1).double()
        
        batch_targets.data[0][0] = inix1 + output[0][3] +c1
        batch_targets.data[0][1] = batch_targets.data[0][0] + output[0][4] +c2
        batch_targets.data[0][2] = batch_targets.data[0][1] + output[0][5] +c3

        batch_targets.data[1][0] = inix2 + output[1][3] +c1
        batch_targets.data[1][1] = batch_targets.data[1][0] + output[1][4] +c2
        batch_targets.data[1][2] = batch_targets.data[1][1] + output[1][5] +c3

        inix1 = batch_targets.data[0][0]
        inix2 = batch_targets.data[1][0]
        
        test_loss += nn.MSELoss()(output, batch_targets)
        nErr += computeErr(output.data)

    nBatches = testX.size(0)/batchSz
    test_loss = test_loss.item()/nBatches
    test_err = nErr/testX.size(0)
    print('TEST SET RESULTS:' + ' ' * 20)
    print('Average loss: {:.4f}'.format(test_loss))
    print('Err: {:.4f}'.format(test_err))

    testW.writerow((epoch, test_loss, test_err))
    testF.flush()

def computeErr(pred):
    batchSz = pred.size(0)
    nsq = int(pred.size(1))
    n = int(np.sqrt(nsq))
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1

    # I = torch.max(pred, 1)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq-1)
        valid *= (x.sum(1) == s)
        return ~valid

    boardCorrect = torch.ones(batchSz).type_as(pred)

    # for j in range(nsq):
    #     # Check the jth row and column.
    #     boardCorrect[invalidGroups(I[:,j,:])] = 0
    #     boardCorrect[invalidGroups(I[:,:,j])] = 0

    #     # Check the jth block.
    #     row, col = n*(j // n), n*(j % n)
    #     M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
    #     boardCorrect[M] = 0

    #     if boardCorrect.sum() == 0:
    #         return batchSz

    # return batchSz-boardCorrect.sum().item()
    return batchSz

if __name__=='__main__':
    main()
