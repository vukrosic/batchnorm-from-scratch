"""Exercise. Freeze every weight at its random starting value. Train only batch norm's two knobs per neuron: gain and shift.

The 20-layer network has 86,410 numbers. Gain and shift are 2,560 of them, about 3 %.
Before you run this: guess the test accuracy after 2,000 steps. Blind guessing is 10 %. The fully trained network reached 96 %.
Then run the file. Then explain how a network of random weights can read digits at all.
Hint: a gain of zero switches a neuron off, and a shift moves the point where its ReLU opens.
(The same test on large image networks: Frankle, Schwab and Morcos, 2020, arXiv 2003.00152.)
"""
from data import split
from net import make, train, accuracy, knobs

STEPS = 2000

if __name__ == '__main__':
    xtr, ytr, xte, yte = split(0)
    net = make(20, 'batch')
    total = sum(k.numel() for k in knobs(net))
    tuned = sum(k.numel() for k in net['gain'] + net['shift'])
    print(f"numbers in the network: {total:,}; gain and shift: {tuned:,} ({100 * tuned / total:.1f} %)")
    for rate in (0.03, 0.1, 0.3):
        net = train(make(20, 'batch'), xtr, ytr, STEPS, rate, only=['gain', 'shift'])
        print(f"only gain and shift, {STEPS:,} steps at rate {rate}: {100 * accuracy(net, xte, yte):.1f} %")
