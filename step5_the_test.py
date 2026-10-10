"""Step 5. The 2018 test: after batch norm, push every neuron to a random new place at every step. Does training still work?"""
import net
import step3_batch_norm
import step4_the_reason
from data import split
from sweep import grid, trained, show, RATES, SCALES


def doubled():
    """The same test with pushes twice as large, to find where it stops working."""
    xtr, ytr, xte, yte = split(0)
    net.PUSH, net.STRETCH = 2 * net.PUSH, 2 * net.STRETCH
    acc = net.accuracy(net.train(net.make(20, 'batch'), xtr, ytr, 400, 0.1, noisy=True), xte, yte)
    net.PUSH, net.STRETCH = net.PUSH / 2, net.STRETCH / 2
    return acc


def run():
    rows = grid('batch', noisy=True)
    w = step4_the_reason.watch('batch', 0.1, noisy=True)
    race = step3_batch_norm.race_noisy(0.1)
    return {'grid': rows, 'trained': trained(rows), 'cells': len(RATES) * len(SCALES), 'best': max(max(r) for r in rows),
            'moved': w['moved'], 'trace': w['trace'], 'race': race, 'doubled': doubled()}


if __name__ == '__main__':
    r = run()
    plain = step4_the_reason.watch(None, 0.03)['moved']
    print(f"with the random pushes, what layer 10 receives moves {r['moved']:.3f} per step ({r["moved"] / plain:.1f} times the plain network)")
    show(r['grid'])
    print(f"it still trains in {r['trained']} of {r['cells']} settings; best {100 * r['best']:.1f} %")
    print(f"at rate 0.1 it first reaches 90 % at step {r['race']['first']}")
    print(f"with pushes twice as large, 400 steps at rate 0.1 reach {100 * r['doubled']:.1f} %")
