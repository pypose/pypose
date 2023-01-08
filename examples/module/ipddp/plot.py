#!/usr/bin/env python3

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import numpy as np

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('save', type=str, default='work/Pendulum-v0')
    args = parser.parse_args()
    fname = os.path.join(args.save, 'pypose losses.csv')
    df = pd.read_csv(fname)

    env_name = list(filter(lambda x: len(x) > 0, args.save.split('\\') ))[-2]

    fig, ax = plt.subplots(figsize=(6,4))
    y = df['im_loss']
    # N = 25
    # y = np.convolve(y, np.full(N, 1./N), mode='valid')
    # x = np.arange(len(y))+N
    x = np.arange(len(y))
    ax.plot(x, y)
    ax.set_ylabel('Imitation Loss')
    ax.set_xlabel('Iteration')
    # ax.set_yscale('log')
    ax.set_ylim((0, None))
    ax.set_title(env_name)
    fig.tight_layout()
    fname = os.path.join(args.save, 'im_loss.png')
    fig.savefig(fname)
    print('Saving to: {}'.format(fname))

    fig, ax = plt.subplots(figsize=(6,4))
    y = df['mse']
    # N = 25
    # y = np.convolve(y, np.full(N, 1./N), mode='valid')
    # x = np.arange(len(y))+N
    x = np.arange(len(y))
    ax.plot(x, y)
    ax.set_ylabel('Cost MSE')
    ax.set_xlabel('Iteration')
    ax.set_ylim((0, None))
    # ax.set_yscale('log')
    ax.set_title(env_name)
    fig.tight_layout()
    fname = os.path.join(args.save, 'mse.png')
    fig.savefig(fname)
    print('Saving to: {}'.format(fname))


if __name__ == "__main__":
    main()