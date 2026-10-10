"""Step 2. Stack 20 layers. The signal shrinks or explodes on the way, and training works only in a narrow window."""
import torch
import torch.nn.functional as F

from data import split
from net import make, forward, peek, train
from sweep import DEPTH, RATES, SCALES, grid, trained, show


def signal(norm, scale):
    """The spread of the numbers going into each layer's neurons, before any training."""
    xtr = split(0)[0]
    keep = []
    with torch.no_grad():
        forward(make(DEPTH, norm, scale), xtr[:256], keep=keep)
    return [float(z.std()) for z in keep]


def run(norm=None):
    rows = grid(norm)
    return {'signal': {str(s): signal(norm, s) for s in SCALES}, 'grid': rows, 'trained': trained(rows),
            'cells': len(RATES) * len(SCALES), 'best': max(max(r) for r in rows), 'rates': RATES, 'scales': SCALES, 'depth': DEPTH}


def start():
    """Before training: the largest of the ten scores and the largest slope of any weight, for three start sizes.
    And for start size 1: how far the numbers at layer 20 drift in 400 training steps."""
    xtr, ytr = split(0)[:2]
    out = {}
    for s in (0.5, 1.0, 2.0):
        net = make(DEPTH, None, s)
        scores = forward(net, xtr[:64])
        slopes = torch.autograd.grad(F.cross_entropy(scores, ytr[:64]), net['W'])
        out[str(s)] = {'score': float(scores.detach().abs().max()), 'slope': max(float(d.abs().max()) for d in slopes), 'first': scores[0].detach().tolist()}
    net = make(DEPTH, None, 1.0, 0)
    before = float(peek(net, xtr[:256])[-1].std())
    net = train(net, xtr, ytr, 400, 0.03, 64, 0, False)
    out['drift'] = {'before': before, 'after': float(peek(net, xtr[:256])[-1].std()), 'rate': 0.03, 'steps': 400}
    return out


if __name__ == '__main__':
    for k, q in start().items():
        print(f'start size x{k}: largest score {q["score"]:.3g}, largest slope {q["slope"]:.3g}' if k != 'drift' else f'start size x1.0, layer 20: spread {q["before"]:.3g} at the start, {q["after"]:.3g} after {q["steps"]} steps')
    r = run()
    for s in (0.5, 1.0, 2.0):
        g = r['signal'][str(s)]
        print(f'start size x{s}: signal at layer 1 = {g[0]:.3g}, layer 10 = {g[9]:.3g}, layer 20 = {g[19]:.3g}')
    show(r['grid'])
    print(f"plain network trains in {r['trained']} of {r['cells']} settings; best {100 * r['best']:.1f} %")
