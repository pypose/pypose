#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from tqdm import tqdm
import time

try: import setGPU
except ImportError: pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import numpy.random as npr

import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import setproctitle

import models

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--nTrials', type=int, default=5)
    # parser.add_argument('--boardSz', type=int, default=2)
    # parser.add_argument('--batchSz', type=int, default=150)
    parser.add_argument('--Qpenalty', type=float, default=0.1)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    setproctitle.setproctitle('bamos.sudoku.prof-sparse')

    print('=== nTrials: {}'.format(args.nTrials))
    print('| {:8s} | {:8s} | {:21s} | {:21s} |'.format(
        'boardSz', 'batchSz', 'dense forward (s)', 'sparse forward (s)'))
    for boardSz in [2,3]:
        with open('data/{}/features.pt'.format(boardSz), 'rb') as f:
            X = torch.load(f)
        with open('data/{}/labels.pt'.format(boardSz), 'rb') as f:
            Y = torch.load(f)
        N, nFeatures = X.size(0), int(np.prod(X.size()[1:]))

        for batchSz in [1, 64, 128]:
            dmodel = models.OptNetEq(boardSz, args.Qpenalty, trueInit=True)
            spmodel = models.SpOptNetEq(boardSz, args.Qpenalty, trueInit=True)
            if args.cuda:
                dmodel = dmodel.cuda()
                spmodel = spmodel.cuda()

            dtimes = []
            sptimes = []
            for i in range(args.nTrials):
                Xbatch = Variable(X[i*batchSz:(i+1)*batchSz])
                Ybatch = Variable(Y[i*batchSz:(i+1)*batchSz])
                if args.cuda:
                    Xbatch = Xbatch.cuda()
                    Ybatch = Ybatch.cuda()

                # Make sure buffers are initialized.
                # dmodel(Xbatch)
                # spmodel(Xbatch)

                start = time.time()
                # dmodel(Xbatch)
                dtimes.append(time.time()-start)

                start = time.time()
                spmodel(Xbatch)
                sptimes.append(time.time()-start)

            print('| {:8d} | {:8d} | {:.2e} +/- {:.2e} | {:.2e} +/- {:.2e} |'.format(
                boardSz, batchSz, np.mean(dtimes), np.std(dtimes),
                np.mean(sptimes), np.std(sptimes)))

if __name__=='__main__':
    main()
