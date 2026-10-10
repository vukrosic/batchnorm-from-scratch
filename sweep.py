"""Train many small networks at once, one per processor core, and take the middle result of three starts."""
import statistics
from multiprocessing import Pool

import torch

from data import split
from net import make, train, accuracy

DEPTH = 20                                    # hidden layers in the deep network
STEPS = 400                                   # training steps per run
RATES = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]    # learning rates: the size of each training step
SCALES = [0.5, 0.7, 1.0, 1.4, 2.0]            # starting size of the weights, 1.0 is the usual one
SEEDS = [0, 1, 2]
GOOD = 0.9                                    # a run "trains" when it is right on 90 % of the test pictures


def one(job):
    """One full training run. Returns the share of test pictures it gets right."""
    torch.set_num_threads(1)
    depth, norm, scale, rate, seed, steps, batch, noisy = job
    xtr, ytr, xte, yte = split(seed)
    net = train(make(depth, norm, scale, seed), xtr, ytr, steps, rate, batch, seed, noisy)
    return accuracy(net, xte, yte)


def many(jobs):
    with Pool() as pool:
        return pool.map(one, jobs)


def middle(settings, depth=DEPTH, steps=STEPS, batch=64):
    """settings: list of (norm, scale, rate, noisy). Returns the middle accuracy of three starts for each."""
    jobs = [(depth, n, s, r, seed, steps, batch, noisy) for (n, s, r, noisy) in settings for seed in SEEDS]
    acc = many(jobs)
    return [statistics.median(acc[i * len(SEEDS):(i + 1) * len(SEEDS)]) for i in range(len(settings))]


def grid(norm, noisy=False):
    """Every learning rate with every starting size. Returns rows of accuracies (one row per starting size)."""
    acc = middle([(norm, s, r, noisy) for s in SCALES for r in RATES])
    return [acc[i * len(RATES):(i + 1) * len(RATES)] for i in range(len(SCALES))]


def trained(rows):
    return sum(a >= GOOD for row in rows for a in row)


def show(rows):
    print('start size   ' + ''.join(f'rate {r:<7}' for r in RATES))
    for s, row in zip(SCALES, rows):
        print(f'  x{s:<9}' + ''.join(f'{100 * a:5.1f} %     ' for a in row))
