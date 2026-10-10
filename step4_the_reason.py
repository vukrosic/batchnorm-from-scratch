"""Step 4. The 2015 paper's reason: without the norm, what each layer receives keeps moving while the layers before it learn."""
import torch

from data import split
from net import make, train, peek, new_noise
from sweep import DEPTH, STEPS

LAYER = 10        # the layer we watch
WATCH = 3         # neurons drawn in the film


def watch(norm, rate, noisy=False):
    """Every step, pass the same 256 pictures through and record what each neuron of layer 10 receives: its mean and spread."""
    xtr, ytr, _, _ = split(0)
    probe, g, means, spreads = xtr[:256], torch.Generator().manual_seed(7), [], []

    def each(step, net):
        z = peek(net, probe, new_noise(DEPTH, g) if noisy else None)[LAYER - 1]
        means.append(z.mean(0))
        spreads.append(z.std(0))

    train(make(DEPTH, norm), xtr, ytr, STEPS, rate, noisy=noisy, each=each)
    m, s = torch.stack(means), torch.stack(spreads)                 # steps x neurons
    scale = s.mean(1, keepdim=True)                                 # the usual spread of a neuron at that step
    moved = float(((m[1:] - m[:-1]).abs() / scale[:-1]).mean())     # average move per step, in units of that spread
    return {'moved': moved, 'trace': (m[:, :WATCH] / scale).T.tolist(), 'rate': rate}


def run():
    return {'plain': watch(None, 0.03), 'batch': watch('batch', 0.1), 'layer': LAYER}


if __name__ == '__main__':
    r = run()
    print(f"what layer {r['layer']} receives moves, per step:")
    print(f"  plain network       {r['plain']['moved']:.4f}")
    print(f"  with batch norm     {r['batch']['moved']:.4f}   ({r['plain']['moved'] / r['batch']['moved']:.0f} times less)")
