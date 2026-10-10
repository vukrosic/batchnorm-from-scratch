"""Step 1. The task and a small network: two hidden layers learn to read handwritten digits."""
from data import X, TRAIN, split
from net import make, train, accuracy
from sweep import middle, STEPS


def run():
    xtr, ytr, xte, yte = split(0)
    curve = []
    train(make(2), xtr, ytr, STEPS, 0.1, each=lambda s, n: curve.append(accuracy(n, xte, yte)) if s % 10 == 0 else None)
    return {'pictures': len(X), 'train': TRAIN, 'test': len(X) - TRAIN, 'guess': 0.1,
            'small': middle([(None, 1.0, 0.1, False)], depth=2)[0], 'curve': curve, 'steps': STEPS, 'rate': 0.1}


if __name__ == '__main__':
    r = run()
    print(f"{r['pictures']} pictures: {r['train']} to learn from, {r['test']} for the test")
    print(f"a blind guess is right {100 * r['guess']:.0f} % of the time")
    print(f"2 hidden layers, {r['steps']} steps at rate {r['rate']}: right on {100 * r['small']:.1f} % of the test pictures")
