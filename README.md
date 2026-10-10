# Batch normalization from scratch

The code from the video **"The Most-Used Trick in Deep Learning Was Explained Wrong"**.

Batch normalization is three lines of code. The 2015 paper that introduced it gave one reason why it works. A 2018 paper tested that reason and it did not hold. This repo builds a small network on a laptop, repeats that test, and measures every step.

```python
mean = z.mean(0)
spread = (z.var(0, unbiased=False) + EPS).sqrt()
return gain * (z - mean) / spread + shift, mean, spread
```

## Run it

```
pip install torch scikit-learn
bash run_all.sh         # the eight checks, the eight steps and the exercise; 3 min 27 s on a MacBook Pro M4
```

The data is 1,797 pictures of handwritten digits, 8 by 8 pixels, that ship inside scikit-learn. Nothing is downloaded. No GPU is needed. The expected output is in `demo-output.txt`.

## The steps

| file | what it does | what it measured |
|---|---|---|
| `net.py` | the network, batch norm, layer norm, the training loop | |
| `step1_task.py` | two hidden layers read the digits | 95.8 % of the test pictures |
| `step2_deep.py` | 20 plain layers, 30 settings (5 start sizes × 6 rates), 400 steps each | trains in 0 of 30 |
| `step3_batch_norm.py` | the three lines after every layer | trains in 18 of 30, best 96.4 % |
| `step4_the_reason.py` | the 2015 reason: how much the input of layer 10 moves per step | plain 0.239, batch norm 0.0012 |
| `step5_the_test.py` | the 2018 test: push every neuron to a random new place at every step | moves 0.332 per step, still trains in 11 of 30 |
| `step6_smoother.py` | the longest step that still lowers the error | plain 0.032, batch norm 0.256 |
| `step7_the_cost.py` | one picture in 200 random batches of 4 | its answer changes in 60 % of them |
| `step8_layer_norm.py` | the same three lines over the neurons of one picture | 0 % changed, but trains in 2 of 30 |
| `exercise.py` | freeze every weight, train only gain and shift | guess first, then run it |

"Trains" means: 90 % or more of the 500 test pictures after 400 steps, as the middle result of three random starts.

## Be careful with these numbers

This is one small network: 20 layers of 64 neurons, 86,410 numbers, trained for 400 steps on small pictures. The papers measured much larger networks. The direction of each result matches the papers. The sizes do not transfer.

Layer norm does badly here because this is a plain stack with no skip connections. In a transformer it sits next to skip connections, and that is a different setup.

## The papers

- Ioffe, Szegedy. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* 2015. arXiv 1502.03167
- Ba, Kiros, Hinton. *Layer Normalization.* 2016. arXiv 1607.06450
- Santurkar, Tsipras, Ilyas, Madry. *How Does Batch Normalization Help Optimization?* 2018. arXiv 1805.11604
- Frankle, Schwab, Morcos. *Training BatchNorm and Only BatchNorm.* 2020. arXiv 2003.00152 (the exercise)

MIT license.
