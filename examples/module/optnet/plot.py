#!/usr/bin/env python3

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import pandas as pd
import numpy as np
import math

import os
import sys
import json
import glob

def main():
    # import sys
    # from IPython.core import ultratb
    # sys.excepthook = ultratb.FormattedTB(mode='Verbose',
    #     color_scheme='Linux', call_pdb=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('workDir', type=str)
    args = parser.parse_args()

    trainF = os.path.join(args.workDir, 'train.csv')
    testF = os.path.join(args.workDir, 'test.csv')

    trainDf = pd.read_csv(trainF, sep=',')
    testDf = pd.read_csv(testF, sep=',')

    plotLoss(trainDf, testDf, args.workDir)

    plotErr(trainDf, testDf, args.workDir)

    initDf = os.path.join(args.workDir, 'D.init')
    if os.path.exists(initDf):
        initD = np.loadtxt(initDf)
        latestD = np.loadtxt(os.path.join(args.workDir, 'D.latest'))
        plotD(initD, latestD, args.workDir)

    loss_fname = os.path.join(args.workDir, 'loss.png')
    err_fname = os.path.join(args.workDir, 'err.png')
    loss_err_fname = os.path.join(args.workDir, 'loss-error.png')
    
    # os.system('convert +append "{}" "{}" "{}"'.format(loss_fname, err_fname, loss_err_fname))
    # print('Created {}'.format(loss_err_fname))

def plotLoss(trainDf, testDf, workDir):
    # fig, ax = plt.subplots(1, 1, figsize=(5,2))
    fig, ax = plt.subplots(1, 1)
    # fig.tight_layout()

    trainEpoch = trainDf['epoch'].values
    trainLoss = trainDf['loss'].values

    N = np.argmax(trainEpoch==1.0)
    trainEpoch = trainEpoch[N-1:]
    trainLoss = np.convolve(trainLoss, np.full(N, 1./N), mode='valid')
    plt.plot(trainEpoch, trainLoss, label='Train')
    if not testDf.empty:
        plt.plot(testDf['epoch'].values, testDf['loss'].values, label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.xlim(xmin=0)
    plt.grid(visible=True, which='major', color='k', linestyle='-')
    plt.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)
    plt.legend()
    # ax.set_yscale('log')
    ax.set_ylim(0, None)
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "loss."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

def plotErr(trainDf, testDf, workDir):
    # fig, ax = plt.subplots(1, 1, figsize=(5,2))
    fig, ax = plt.subplots(1, 1)
    # fig.tight_layout()

    trainEpoch = trainDf['epoch'].values
    trainLoss = trainDf['err'].values

    N = np.argmax(trainEpoch==1.0)
    trainEpoch = trainEpoch[N-1:]
    trainLoss = np.convolve(trainLoss, np.full(N, 1./N), mode='valid')
    plt.plot(trainEpoch, trainLoss, label='Train')
    if not testDf.empty:
        plt.plot(testDf['epoch'].values, testDf['err'].values, label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.xlim(xmin=0)
    plt.grid(visible=True, which='major', color='k', linestyle='-')
    plt.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)
    plt.legend()
    # ax.set_yscale('log')
    ax.set_ylim(0, None)
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "err."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

def plotD(initD, latestD, workDir):
    def p(D, fname):
        plt.clf()
        lim = max(np.abs(np.min(D)), np.abs(np.max(D)))
        clim = (-lim, lim)
        plt.imshow(D, cmap='bwr', interpolation='nearest', clim=clim)
        plt.colorbar()
        plt.savefig(os.path.join(workDir, fname))

    p(initD, 'initD.png')
    p(latestD, 'latestD.png')

    latestDs = latestD**6
    latestDs = latestDs/np.sum(latestDs, axis=1)[:,None]
    I = np.argsort(latestDs.dot(np.arange(latestDs.shape[1])))
    latestDs = latestD[I]
    initDs = initD[I]

    p(initDs, 'initD_sorted.png')
    p(latestDs, 'latestD_sorted.png')

    # Dcombined = np.concatenate((initDs, np.zeros((initD.shape[0], 10)), latestDs), axis=1)
    # p(Dcombined, 'Dcombined.png')

if __name__ == '__main__':
    main()
