"""Step 3. Add batch norm after every layer. The signal stays the same size, and training works almost everywhere."""
import step2_deep
from data import split
from net import make, train, accuracy
from sweep import DEPTH, STEPS, show


def race(norm, rate, steps=STEPS):
    """Test accuracy every 10 steps, and the first step at which it reaches 90 %."""
    xtr, ytr, xte, yte = split(0)
    curve = []
    train(make(DEPTH, norm), xtr, ytr, steps, rate, each=lambda s, n: curve.append(accuracy(n, xte, yte)) if s % 10 == 0 else None)
    first = next((10 * i for i, a in enumerate(curve) if a >= 0.9), None)
    return {'curve': curve, 'first': first, 'rate': rate, 'steps': steps, 'end': curve[-1]}


def race_noisy(rate):
    """The same race for batch norm with the random pushes of step 5. The test itself runs without pushes."""
    xtr, ytr, xte, yte = split(0)
    curve = []
    train(make(DEPTH, 'batch'), xtr, ytr, STEPS, rate, noisy=True, each=lambda s, n: curve.append(accuracy(n, xte, yte)) if s % 10 == 0 else None)
    return {'curve': curve, 'first': next((10 * i for i, a in enumerate(curve) if a >= 0.9), None), 'rate': rate, 'steps': STEPS, 'end': curve[-1]}


def run():
    r = step2_deep.run('batch')
    r['race'] = {'plain': race(None, 0.03, 10 * STEPS), 'batch': race('batch', 0.1)}   # each at its best setting; plain gets 10 times the steps
    return r


if __name__ == '__main__':
    r = run()
    for s in (0.5, 1.0, 2.0):
        g = r['signal'][str(s)]
        print(f'start size x{s}: signal at layer 1 = {g[0]:.3g}, layer 10 = {g[9]:.3g}, layer 20 = {g[19]:.3g}')
    show(r['grid'])
    print(f"with batch norm it trains in {r['trained']} of {r['cells']} settings; best {100 * r['best']:.1f} %")
    for k, v in r['race'].items():
        print(f"{k:5} at rate {v['rate']}, {v['steps']} steps: first reaches 90 % at step {v['first']}, ends at {100 * v['end']:.1f} %")
