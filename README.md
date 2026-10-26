# CTM with a Clocked Event Memory

Code for the design in the thesis *Remembering When* (`~/Downloads/Remembering_When_CEM_Thesis.pdf`).

A Continuous Thought Machine (CTM) gets a diary. At the end of every input event it writes one page. Each
page has a stamp from a designed clock, a sparse sketch of what the event looked like, and a learned value.
On every tick, four fixed heads read the diary (by number, next page, previous page, by content), and the
read is added to what the CTM sees. At the start the read is exactly zero, so training starts from the
plain CTM.

## Files

| File | What it does | Thesis |
|---|---|---|
| `cem/clock.py` | Stamps, the trust weights designed by a linear program, turning the hands | §3.2, eqs. 3–5 and 7 |
| `cem/sketch.py` | The sparse sketch that the content head compares | §3.4, eq. 8 |
| `cem/memory.py` | The diary: writing a page, the four heads, the spotlight, the read, crossing off | §3, eqs. 2, 6, 9, 10 |
| `cem/model.py` | The upstream Q&A-MNIST CTM with the diary attached | Figure 1 |
| `cem/task.py` | Q&A-MNIST episodes, with one-hot digits or real MNIST images | §2.2 |
| `cem/upstream.py` | Makes the pinned upstream CTM code importable | |
| `train.py` | Trains and evaluates the CTM with or without the diary | §8 |
| `test_cem.py` | Checks the code against the thesis's results | §3–§7 |
| `verify.py` | Checks the thesis's predictions against the trained models, and runs the extra checks | "What to test next" |

## Setup

    ./setup.sh

This clones the upstream CTM into `third_party/` at the pinned commit and installs the packages. The
upstream code is already on this machine, so you can link to it instead of cloning:

    mkdir -p third_party
    ln -s ~/ctm-rm/third_party/continuous-thought-machines third_party/continuous-thought-machines
    pip install -r requirements.txt

## Test

    python -m pytest test_cem.py

26 checks, about twenty seconds. Each one tests a result of the thesis against the code: the clock's margin, finding
and flipping pages, sketches, edits, what a page holds, the silent start (with one-hot digits and with images), the
order in which learning starts. The others check that without the diary the model computes exactly what the upstream
code computes, the latest-page head, the MNIST images, their checksum and the pretrained backbone, and the probes
and ablations in `verify.py`.

## Run

    python train.py                   # the CTM with the diary, seed 0
    python train.py --no-diary        # the plain CTM: same seed, same starting weights, same data
    python train.py --seed 3 --device cuda

    python train.py --no-diary --d-model 107    # a plain CTM with as many parameters as the diary's (51,164 vs 50,925)
    python train.py --latest-head               # the diary with a fifth head that reads the latest page
    python train.py --images                    # real MNIST images: training images to train, test images to evaluate
    python train.py --images --pretrain-backbone --steps 16000    # a frozen, pretrained image backbone; 4x the training

`--no-diary` reproduces the "ctm" runs of `~/ctm` (thesis Box 1) exactly: for seed 0, all 36 accuracy cells and the
whole loss curve match `~/ctm/results/primary/ctm_s0.json`. Training uses one CPU thread for that reason.

Each run prints accuracy by number of digits (1–4 as in training, and 5, 6, 8, 12, 16 for length). It saves
`results/<variant>_seed<seed>.json` and the trained weights next to it. The variant is `ctm` or `ctm_with_diary`,
then `_latest`, `_d<neurons>` and `_mnist` for the options used.

`--images` needs Keras's copy of MNIST at `~/.keras/datasets/mnist.npz`. The file is checked against Keras's
checksum and read without pickle. torchvision is not needed. Trained from scratch for the same 4000 steps, most
seeds never learn to name even a single digit, with or without the diary. `--pretrain-backbone` therefore first
trains the backbone as a digit classifier on the training images (1000 steps; it then names about 0.94 of the test
digits) and then freezes it. Both variants of a seed get the same backbone. Left to train along with the CTM at the
Q&A learning rate, the backbone lost a fifth of what it had learned (a fresh linear read-out named 0.83 of the test
digits instead of 0.95), and the CTM still named only a third to two thirds of single digits after 4000 steps.

`--device mps` works: on a trained diary, MPS and CPU give the same answers, and every gradient agrees to 2e-6. A
full run on MPS is not bit-for-bit the CPU run, but lands close to it (seed 0, two digits: 0.660 against 0.643).
For a model this small it is about eight times slower than one CPU thread (45 minutes against 6).

## Check the thesis's predictions

Train both variants for seeds 0–19, eight runs at a time (each run uses one CPU thread), then check:

    seq 0 19 | xargs -P 8 -I{} python3 train.py --seed {}
    seq 0 19 | xargs -P 8 -I{} python3 train.py --seed {} --no-diary
    python verify.py

`verify.py` tests the five predictions of the thesis's "What to test next" section. It uses the saved accuracies,
and probes the trained diaries with pure-recall questions of 2 to 48 digits: does a by-number read decode to the
asked digit, and where do the wrong answers land? It prints a verdict for each prediction and saves
`results/verification.json`. It checks eight seeds at a time (`--workers`); each uses up to about 2.5 GB, for
the 48-digit probes.

For the extra checks, train these too (an MNIST run takes three to five times as long as a symbolic one):

    seq 0 19 | xargs -P 8 -I{} python3 train.py --seed {} --no-diary --d-model 107
    seq 0 19 | xargs -P 8 -I{} python3 train.py --seed {} --latest-head
    seq 0 19 | xargs -P 8 -I{} python3 train.py --seed {} --images
    seq 0 19 | xargs -P 8 -I{} python3 train.py --seed {} --no-diary --images
    seq 0 4 | xargs -P 8 -I{} python3 train.py --seed {} --images --pretrain-backbone --steps 16000
    seq 0 4 | xargs -P 8 -I{} python3 train.py --seed {} --no-diary --images --pretrain-backbone --steps 16000

`verify.py` then also compares, paired by seed, the diary with the plain CTM of the same size, the diary with and
without the latest-page head, and the diary with the plain CTM on real images (from scratch, and with the frozen
pretrained backbone and 4x the training, on 5 seeds). Each comparison uses the seeds both variants have. It silences
each head of the trained diaries in turn and reports the accuracy lost. Variants that have not been trained are
skipped.
