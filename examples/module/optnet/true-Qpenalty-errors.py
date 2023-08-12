#!/usr/bin/env python3

import torch
from torch.autograd import Variable

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import models
from train import computeErr

batchSz = 128

boards = {}
for boardSz in (2,3):
    with open('data/{}/features.pt'.format(boardSz), 'rb') as f:
        unsolvedBoards = Variable(torch.load(f).cuda()[:,:,:,:])
        nBoards = unsolvedBoards.size(0)
    with open('data/{}/labels.pt'.format(boardSz), 'rb') as f:
        solvedBoards = Variable(torch.load(f).cuda()[:nBoards,:,:,:])
    boards[boardSz] = (unsolvedBoards, solvedBoards)

nBatches = nBoards//batchSz
results = {}
startIdx = 0

ranges = {
    2: np.linspace(0.1, 2.0, num=11),
    3: np.linspace(0.1, 1.0, num=10)
}

for i in range(nBatches):
    nSeen = (i+1)*batchSz
    print('=== {} Boards Seen ==='.format(nSeen))

    for boardSz in (2,3):
        unsolvedBoards, solvedBoards = boards[boardSz]

        print('--- Board Sz: {} ---'.format(boardSz))
        print('| {:15s} | {:15s} | {:15s} |'.format('Qpenalty', '% Boards Wrong', '# Blanks Wrong'))

        for j,Qpenalty in enumerate(ranges[boardSz]):
            model = models.OptNetEq(boardSz, Qpenalty, trueInit=True).cuda()
            X_batch = unsolvedBoards[startIdx:startIdx+batchSz]
            Y_batch = solvedBoards[startIdx:startIdx+batchSz]
            preds = model(X_batch).data
            err = computeErr(preds)

            # nWrong is not an exact metric because a board might have multiple solutions.
            predBoards = torch.max(preds, 3)[1].squeeze().view(batchSz, -1)
            trueBoards = torch.max(Y_batch.data, 3)[1].squeeze().view(batchSz, -1)
            nWrong = ((predBoards-trueBoards).abs().cpu().numpy() > 1e-7).sum(axis=1)

            results_key = (boardSz, j)
            if results_key not in results:
                results_j = {'err': err, 'nWrong': nWrong}
                results[results_key] = results_j
            else:
                results_j = results[results_key]
                results_j['err'] += err
                results_j['nWrong'] = np.concatenate((results_j['nWrong'], nWrong))

            err = results_j['err']/(batchSz*(i+1))
            nWrong = np.mean(results_j['nWrong'])
            print('| {:15f} | {:15f} | {:15f} |'.format(Qpenalty, err, nWrong))

    print('='*50)
    print('\n\n')

    startIdx += batchSz
