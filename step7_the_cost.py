"""Step 7. The cost: with batch norm, the answer for one picture depends on the other pictures in its batch."""
import torch

from data import split
from net import make, train, forward, accuracy
from sweep import DEPTH, STEPS, middle

BATCHES = [2, 4, 8, 16, 32, 64]
TRIES = 200


def mates(net, xte, size, g, picture=0):
    """Put one test picture into 200 random batches. How often does the network's answer for it change?"""
    answers, sure = [], []
    for _ in range(TRIES):
        others = torch.randint(1, len(xte), (size - 1,), generator=g)
        saved = [m.clone() for m in net['mean']], [m.clone() for m in net['spread']]
        with torch.no_grad():
            out = forward(net, torch.cat([xte[picture:picture + 1], xte[others]]))[0].softmax(0)
        net['mean'], net['spread'] = saved
        answers.append(int(out.argmax()))
        sure.append(float(out.max()))
    usual = max(set(answers), key=answers.count)
    return {'changed': sum(a != usual for a in answers) / TRIES, 'low': min(sure), 'high': max(sure)}


def train_mode(net, xte, yte, size):
    """Accuracy when the network keeps using the batch's own numbers at test time, with test batches of this size."""
    right = 0
    for i in range(0, len(xte) - size + 1, size):
        saved = [m.clone() for m in net['mean']], [m.clone() for m in net['spread']]
        with torch.no_grad():
            right += int((forward(net, xte[i:i + size]).argmax(1) == yte[i:i + size]).sum())
        net['mean'], net['spread'] = saved
    return right / (len(xte) // size * size)


def run(norm='batch', rate=0.1):
    xtr, ytr, xte, yte = split(0)
    net, g = train(make(DEPTH, norm), xtr, ytr, STEPS, rate), torch.Generator().manual_seed(3)
    return {'saved': accuracy(net, xte, yte),
            'mates': {size: mates(net, xte, size, g) for size in (4, 64)},
            'wobbly': sum(mates(net, xte, 8, g, p)['changed'] > 0 for p in range(100)) / 100,
            'train_mode': {size: train_mode(net, xte, yte, size) for size in [1] + BATCHES},
            'small': {b: middle([(norm, 1.0, rate, False)], batch=b)[0] for b in BATCHES},
            'rate': rate, 'batches': BATCHES}


def show(r):
    print(f"tested with the saved averages: {100 * r['saved']:.1f} %")
    for size, m in r['mates'].items():
        print(f"one picture in 200 random batches of {size}: answer changes in {100 * m['changed']:.1f} % of them, sureness from {m['low']:.2f} to {m['high']:.2f}")
    print(f"of 100 test pictures, {100 * r['wobbly']:.0f} get more than one answer in batches of 8")
    print("tested with the batch's own numbers:   " + "   ".join(f"{s}: {100 * a:.1f} %" for s, a in r['train_mode'].items()))
    print("trained with batches of this size:     " + "   ".join(f"{s}: {100 * a:.1f} %" for s, a in r['small'].items()))


if __name__ == '__main__':
    show(run())
