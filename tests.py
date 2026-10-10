"""Eight checks that the pieces do what the film says. Run: python3 tests.py"""
import torch

from data import split
from net import make, train, forward, accuracy, batch_norm, layer_norm

xtr, ytr, xte, yte = split(0)
z = torch.randn(64, 8) * 5 + 3
one, zero = torch.ones(8), torch.zeros(8)


def check(name, ok):
    print(('ok    ' if ok else 'FAIL  ') + name)
    assert ok, name


out, _, _ = batch_norm(z, one, zero)
check('batch norm gives every neuron mean 0 and spread 1 over the batch', out.mean(0).abs().max() < 1e-5 and (out.std(0, unbiased=False) - 1).abs().max() < 1e-3)

out, _, _ = batch_norm(z, 2 * one, zero + 0.5)
check('gain and shift set the new spread and mean', (out.mean(0) - 0.5).abs().max() < 1e-5 and (out.std(0, unbiased=False) - 2).abs().max() < 1e-3)

out = layer_norm(z, one, zero)
check('layer norm gives every picture mean 0 and spread 1 over its neurons', out.mean(1).abs().max() < 1e-5 and (out.std(1, unbiased=False) - 1).abs().max() < 1e-3)

net = make(20, 'batch')
before = forward(net, xtr[:64]).detach()
with torch.no_grad():
    net['W'][10] *= 10
    net['b'][10] *= 10
check('with batch norm, ten times larger weights give the same output', (forward(net, xtr[:64]) - before).abs().max() < 1e-3)

net = make(20, 'layer')
alone, together = forward(net, xte[:1])[0], forward(net, xte[:64])[0]
check('with layer norm, a picture gets the same answer alone or in a batch', (alone - together).abs().max() < 1e-4)

net = make(20, 'batch')
check('batch norm on one picture alone erases the picture', (forward(net, xte[:1]) - forward(net, xte[1:2])).abs().max() < 1e-4)

alone, together = forward(net, xte[:1], train=False)[0], forward(net, xte[:64], train=False)[0]
check('with the saved averages, batch norm no longer depends on the batch', (alone - together).abs().max() < 1e-4)

check('the two-layer network learns the digits', accuracy(train(make(2), xtr, ytr, 400, 0.1), xte, yte) > 0.9)
print('8 of 8 checks pass')
