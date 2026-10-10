"""Step 8. Layer norm: the same three lines, turned sideways. Each picture is measured against its own neurons, not against its batch."""
import step2_deep
import step7_the_cost
from sweep import grid, trained, show, RATES, SCALES


def run():
    rows = grid('layer')
    best = max(rows[SCALES.index(1.0)])
    rate = RATES[rows[SCALES.index(1.0)].index(best)]
    return {'signal': step2_deep.signal('layer', 1.0), 'grid': rows, 'trained': trained(rows), 'cells': len(RATES) * len(SCALES),
            'best': max(max(r) for r in rows), 'rate': rate, 'cost': step7_the_cost.run('layer', rate)}


if __name__ == '__main__':
    r = run()
    show(r['grid'])
    print(f"trains in {r['trained']} of {r['cells']} settings; best {100 * r['best']:.1f} %; the cost test at rate {r['rate']}:")
    step7_the_cost.show(r['cost'])
