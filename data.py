"""The task: 1,797 small pictures of handwritten digits, 8 by 8 pixels each. They ship inside scikit-learn."""
import torch
from sklearn.datasets import load_digits

_d = load_digits()
X = torch.tensor(_d.data / 16.0, dtype=torch.float32)   # 64 numbers per picture, each from 0 (white) to 1 (black)
Y = torch.tensor(_d.target)                             # the digit a person wrote, 0 to 9
TRAIN = 1297                                            # pictures to learn from; the other 500 are kept for the test


def split(seed=0):
    """Shuffle, then cut into pictures to learn from and pictures for the test."""
    p = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    return X[p[:TRAIN]], Y[p[:TRAIN]], X[p[TRAIN:]], Y[p[TRAIN:]]
