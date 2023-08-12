#!/usr/bin/env python3
#
# Some portions from: https://www.ocf.berkeley.edu/~arel/sudoku/main.html

import argparse
import numpy as np
import numpy.random as npr
import torch

from tqdm import tqdm

import os, sys
import shutil

import random, copy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boardSz', type=int, default=2)
    parser.add_argument('--nSamples', type=int, default=10000)
    parser.add_argument('--data', type=str, default='data')
    args = parser.parse_args()

    npr.seed(0)

    save = os.path.join(args.data, str(args.boardSz))
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)

    X = []
    Y = []
    for i in tqdm(range(args.nSamples)):
        Xi, Yi = sample(args)
        X.append(Xi)
        Y.append(Yi)

    X = np.array(X)
    Y = np.array(Y)

    for loc,arr in (('features.pt', X), ('labels.pt', Y)):
        fname = os.path.join(save, loc)
        with open(fname, 'wb') as f:
            torch.save(torch.Tensor(arr), f)
        print("Created {}".format(fname))

def sample(args):
    solution = construct_puzzle_solution(args.boardSz)
    Nsq = args.boardSz*args.boardSz
    nKeep = npr.randint(0, Nsq)
    board, nKept = pluck(copy.deepcopy(solution), nKeep)
    solution = toOneHot(solution)
    board = toOneHot(board)
    return board, solution

def toOneHot(X):
    X = np.array(X)
    Nsq = X.shape[0]
    Y = np.zeros((Nsq, Nsq, Nsq))
    for i in range(1,Nsq+1):
        Y[:,:,i-1][X == i] = 1.0
    return Y

def construct_puzzle_solution(N):
    """
    Randomly arrange numbers in a grid while making all rows, columns and
    squares (sub-grids) contain the numbers 1 through Nsq.

    For example, "sample" (above) could be the output of this function. """
    # Loop until we're able to fill all N^4 cells with numbers, while
    # satisfying the constraints above.
    Nsq = N*N
    while True:
        try:
            puzzle  = [[0]*Nsq for i in range(Nsq)] # start with blank puzzle
            rows    = [set(range(1,Nsq+1)) for i in range(Nsq)] # set of available
            columns = [set(range(1,Nsq+1)) for i in range(Nsq)] #   numbers for each
            squares = [set(range(1,Nsq+1)) for i in range(Nsq)] #   row, column and square
            for i in range(Nsq):
                for j in range(Nsq):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i].intersection(columns[j]).intersection(
                        squares[(i//N)*N + j//N])
                    choice  = random.choice(list(choices))

                    puzzle[i][j] = choice

                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i//N)*N + j//N].discard(choice)

            # success! every cell is filled.
            return puzzle

        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass

def pluck(puzzle, nKeep=0):
    """
    Randomly pluck out K cells (numbers) from the solved puzzle grid, ensuring that any
    plucked number can still be deduced from the remaining cells.

    For deduction to be possible, each other cell in the plucked number's row, column,
    or square must not be able to contain that number. """

    Nsq = len(puzzle)
    N = int(np.sqrt(Nsq))


    def canBeA(puz, i, j, c):
        """
        Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
        in cell "c"? """
        v = puz[c//Nsq][c%Nsq]
        if puz[i][j] == v: return True
        if puz[i][j] in range(1,Nsq+1): return False

        for m in range(Nsq): # test row, col, square
            # if not the cell itself, and the mth cell of the group contains the value v, then "no"
            if not (m==c//Nsq and j==c%Nsq) and puz[m][j] == v: return False
            if not (i==c//Nsq and m==c%Nsq) and puz[i][m] == v: return False
            if not ((i//N)*N + m//N==c//Nsq and (j//N)*N + m%N==c%Nsq) \
               and puz[(i//N)*N + m//N][(j//N)*N + m%N] == v:
                return False

        return True


    """
    starts with a set of all N^4 cells, and tries to remove one (randomly) at a time
    but not before checking that the cell can still be deduced from the remaining cells. """
    cells     = set(range(Nsq*Nsq))
    cellsleft = cells.copy()
    while len(cells) > nKeep and len(cellsleft):
        cell = random.choice(list(cellsleft)) # choose a cell from ones we haven't tried
        cellsleft.discard(cell) # record that we are trying this cell

        # row, col and square record whether another cell in those groups could also take
        # on the value we are trying to pluck. (If another cell can, then we can't use the
        # group to deduce this value.) If all three groups are True, then we cannot pluck
        # this cell and must try another one.
        row = col = square = False

        for i in range(Nsq):
            if i != cell//Nsq:
                if canBeA(puzzle, i, cell%Nsq, cell): row = True
            if i != cell%Nsq:
                if canBeA(puzzle, cell//Nsq, i, cell): col = True
            if not (((cell//Nsq)/N)*N + i//N == cell//Nsq and ((cell//Nsq)%N)*N + i%N == cell%Nsq):
                if canBeA(puzzle, ((cell//Nsq)//N)*N + i//N,
                          ((cell//Nsq)%N)*N + i%N, cell): square = True

        if row and col and square:
            continue # could not pluck this cell, try again.
        else:
            # this is a pluckable cell!
            puzzle[cell//Nsq][cell%Nsq] = 0 # 0 denotes a blank cell
            cells.discard(cell) # remove from the set of visible cells (pluck it)
            # we don't need to reset "cellsleft" because if a cell was not pluckable
            # earlier, then it will still not be pluckable now (with less information
            # on the board).

    # This is the puzzle we found, in all its glory.
    return (puzzle, len(cells))

if __name__=='__main__':
    main()
