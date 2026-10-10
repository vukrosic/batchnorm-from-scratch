"""A plain deep network, and the same network with batch norm or layer norm after every layer."""
import torch
import torch.nn.functional as F

INPUTS, WIDTH, CLASSES = 64, 64, 10
EPS = 1e-5


def batch_norm(z, gain, shift):
    """The 2015 method. For every neuron: measure its mean and spread over the batch, then remove them."""
    mean = z.mean(0)
    spread = (z.var(0, unbiased=False) + EPS).sqrt()
    return gain * (z - mean) / spread + shift, mean, spread


def layer_norm(z, gain, shift):
    """The 2016 method. For every picture: measure the mean and spread over the neurons of the layer."""
    mean = z.mean(1, keepdim=True)
    spread = (z.var(1, unbiased=False, keepdim=True) + EPS).sqrt()
    return gain * (z - mean) / spread + shift


def make(depth, norm=None, scale=1.0, seed=0):
    """Starting weights for `depth` hidden layers. scale 1.0 is the usual starting size for this kind of layer."""
    g = torch.Generator().manual_seed(seed)
    sizes = [INPUTS] + [WIDTH] * depth + [CLASSES]
    net = {'norm': norm, 'W': [], 'b': [], 'gain': [], 'shift': [], 'mean': [], 'spread': []}
    for a, c in zip(sizes[:-1], sizes[1:]):
        net['W'].append((torch.randn(a, c, generator=g) * scale * (2 / a) ** 0.5).requires_grad_())
        net['b'].append(torch.zeros(c, requires_grad=True))
    for _ in range(depth):
        net['gain'].append(torch.ones(WIDTH, requires_grad=True))     # learned: the spread the layer wants
        net['shift'].append(torch.zeros(WIDTH, requires_grad=True))   # learned: the mean the layer wants
        net['mean'].append(torch.zeros(WIDTH))                        # running averages, used after training
        net['spread'].append(torch.ones(WIDTH))
    return net


def knobs(net):
    return net['W'] + net['b'] + (net['gain'] + net['shift'] if net['norm'] else [])


def forward(net, x, train=True, noise=None, keep=None):
    """Run pictures through the network. `keep` collects what goes into each layer's neurons."""
    for i in range(len(net['W']) - 1):
        z = x @ net['W'][i] + net['b'][i]
        if net['norm'] == 'batch' and train:
            z, mean, spread = batch_norm(z, net['gain'][i], net['shift'][i])
            net['mean'][i] = 0.9 * net['mean'][i] + 0.1 * mean.detach()
            net['spread'][i] = 0.9 * net['spread'][i] + 0.1 * spread.detach()
        elif net['norm'] == 'batch':
            z = net['gain'][i] * (z - net['mean'][i]) / net['spread'][i] + net['shift'][i]
        elif net['norm'] == 'layer':
            z = layer_norm(z, net['gain'][i], net['shift'][i])
        if noise is not None:                          # the 2018 test: push every neuron somewhere else, new each step
            z = z * noise[i][0] + noise[i][1]
        if keep is not None:
            keep.append(z.detach())
        x = F.relu(z)
    return x @ net['W'][-1] + net['b'][-1]


PUSH, STRETCH = 0.5, 0.25     # the random pushes of the 2018 test: shift up to 0.5 either way, stretch from 0.75 to 1.25


def new_noise(depth, g):
    """For each layer and neuron, a random stretch and a random shift."""
    return [(1 + STRETCH * (2 * torch.rand(WIDTH, generator=g) - 1), PUSH * (2 * torch.rand(WIDTH, generator=g) - 1)) for _ in range(depth)]


def peek(net, x, noise=None):
    """What goes into every layer's neurons for these pictures, without changing the network."""
    saved, keep = ([m.clone() for m in net['mean']], [m.clone() for m in net['spread']]), []
    with torch.no_grad():
        forward(net, x, noise=noise, keep=keep)
    net['mean'], net['spread'] = saved
    return keep


def accuracy(net, x, y, train=False):
    with torch.no_grad():
        out = forward(net, x, train=train)
    return float((out.argmax(1) == y).float().mean()) if torch.isfinite(out).all() else 0.1


def train(net, xtr, ytr, steps=400, rate=0.05, batch=64, seed=0, noisy=False, each=None, only=None):
    """Plain gradient descent on batches. `each(step, net)` is called before every step, for measurements."""
    g = torch.Generator().manual_seed(1000 + seed)
    depth = len(net['W']) - 1
    learn = knobs(net) if only is None else [k for name in only for k in net[name]]     # `only` trains just these groups
    for step in range(steps):
        if each:
            each(step, net)
        pick = torch.randint(0, len(xtr), (batch,), generator=g)
        noise = new_noise(depth, g) if noisy else None
        loss = F.cross_entropy(forward(net, xtr[pick], noise=noise), ytr[pick])
        if not torch.isfinite(loss):
            return net                                  # the numbers blew up: training has failed
        grads = torch.autograd.grad(loss, learn)
        with torch.no_grad():
            for k, d in zip(learn, grads):
                k -= rate * d
    return net
