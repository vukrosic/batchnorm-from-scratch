"""Step 6. The 2018 answer: with batch norm the slope stays trustworthy for a longer step, so bigger steps are safe."""
import torch
import torch.nn.functional as F

from data import split
from net import make, train, forward, knobs, new_noise
from sweep import DEPTH, STEPS

SIZES = [0.001 * 2 ** k for k in range(15)]          # step sizes to try, from 0.001 to about 16
EVERY = 20                                           # look around every 20 training steps


def loss_and_slope(net, x, y, noise=None):
    saved = [m.clone() for m in net['mean']], [m.clone() for m in net['spread']]
    loss = F.cross_entropy(forward(net, x, noise=noise), y)
    slope = torch.autograd.grad(loss, knobs(net))
    net['mean'], net['spread'] = saved
    return float(loss.detach()), slope


def moved(net, slope, size):
    """Move every knob downhill by `size` times its slope (a negative size moves back)."""
    with torch.no_grad():
        for k, d in zip(knobs(net), slope):
            k -= size * d


def look(net, x, y, rate, noise=None):
    """From where the network stands: how large a step still lowers the loss, and does the slope after one step agree with the slope before?"""
    here, slope = loss_and_slope(net, x, y, noise)
    safe = 0.0
    for size in SIZES:
        moved(net, slope, size)
        there, _ = loss_and_slope(net, x, y, noise)
        moved(net, slope, -size)
        if not there < here:
            break
        safe = size
    moved(net, slope, rate)
    _, after = loss_and_slope(net, x, y, noise)
    moved(net, slope, -rate)
    a, b = torch.cat([d.flatten() for d in slope]), torch.cat([d.flatten() for d in after])
    agree = float(a @ b / (a.norm() * b.norm() + 1e-30))
    return safe, agree


def survey(norm, rate, step=0.1, noisy=False):
    """Train as usual and look around every 20 steps. `step` is the test step for the agreement check."""
    xtr, ytr, _, _ = split(0)
    px, py, safe, agree, g = xtr[:256], ytr[:256], [], [], torch.Generator().manual_seed(7)

    def each(s, net):
        if s % EVERY == 0:
            a, b = look(net, px, py, step, new_noise(DEPTH, g) if noisy else None)
            safe.append(a)
            agree.append(b)

    train(make(DEPTH, norm), xtr, ytr, STEPS, rate, noisy=noisy, each=each)
    mid = lambda v: sorted(v)[len(v) // 2]
    return {'safe': safe, 'agree': agree, 'safe_mid': mid(safe), 'agree_mid': mid(agree), 'rate': rate, 'step': step}


def bigger_weights(norm, times=10.0):
    """Multiply one middle layer's weights by ten. How much does the output change, and what happens to that layer's slope?"""
    xtr, ytr, _, _ = split(0)
    x, y, net, i = xtr[:256], ytr[:256], make(DEPTH, norm), DEPTH // 2
    out = forward(net, x).detach()
    _, slope = loss_and_slope(net, x, y)
    with torch.no_grad():
        net['W'][i] *= times
        net['b'][i] *= times
    out2 = forward(net, x).detach()
    _, slope2 = loss_and_slope(net, x, y)
    return {'change': float((out2 - out).abs().max() / out.abs().max()), 'slope': float(slope2[i].norm() / slope[i].norm())}


def run():
    return {'plain': survey(None, 0.03), 'batch': survey('batch', 0.03), 'batch_fast': survey('batch', 0.1), 'pushed': survey('batch', 0.1, noisy=True),
            'weights': {'plain': bigger_weights(None), 'batch': bigger_weights('batch')}, 'sizes': SIZES, 'every': EVERY}


if __name__ == '__main__':
    r = run()
    for name, k in (('plain network, rate 0.03', 'plain'), ('batch norm, rate 0.03', 'batch'), ('batch norm, rate 0.1', 'batch_fast'), ('with random pushes, 0.1', 'pushed')):
        print(f"{name:26s} largest step that still lowers the loss: {r[k]['safe_mid']:.3f}   slope agreement after a step of 0.1: {r[k]['agree_mid']:+.2f}")
    for name in ('plain', 'batch'):
        w = r['weights'][name]
        print(f"{name:6s} one layer's weights times ten: output changes by {100 * w['change']:.1f} %, that layer's slope becomes {w['slope']:.2f} times as large")
