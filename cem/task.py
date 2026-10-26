"""
Symbolic Q&A-MNIST episodes (thesis Section 2.2), in the upstream input format.

An episode shows n_digits digits, one at a time, for R ticks each. Then the question comes in parts of R
ticks each: index, operator, index, operator, ..., index. Then come R answer ticks with no input. The
answer is the running result modulo 10:

    answer = ( ((d[i0] op1 d[i1]) op2 d[i2]) ... ) mod 10

For example, the digits 7, 2, 9 and the question "index 2, plus, index 0" ask for the third digit plus the
first: 9 + 7 = 16, so the answer is 6. With no operators the task is pure recall of one digit.

Digits are one-hot vectors, which separates memory from perception, or real MNIST images (MNISTImages).
Question tokens use the upstream codes: an index is its number (0, 1, 2, ...), plus is -1 and minus is -2.
"""
import hashlib
import os

import numpy as np
import torch

PLUS = -1
MINUS = -2

MNIST_FILE = os.path.expanduser("~/.keras/datasets/mnist.npz")
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"   # the file Keras downloads
MNIST_MEAN = 0.1307                    # the usual normalisation, as in our earlier runs
MNIST_STD = 0.3081


class MNISTImages:
    """
    Real MNIST images grouped by digit, for showing the digits as images instead of one-hot vectors.

    split:   "train" or "test"
    path:    Keras's copy of MNIST: an .npz file of uint8 arrays x_train, y_train, x_test and y_test
    sha256:  the file's checksum. The file was downloaded, so it is checked before use, and it is read
             without pickle, so it cannot run code.
    """

    def __init__(self, split: str, path: str = MNIST_FILE, sha256: str = MNIST_SHA256):
        with open(path, "rb") as file:
            checksum = hashlib.sha256(file.read()).hexdigest()
        if checksum != sha256:
            raise ValueError(f"{path} is not the expected MNIST file: its sha256 is {checksum}")

        with np.load(path, allow_pickle=False) as data:
            pixels = torch.from_numpy(data[f"x_{split}"])          # (N, 28, 28) uint8
            labels = torch.from_numpy(data[f"y_{split}"]).long()   # (N,)

        images = pixels.float().div(255.0).sub(MNIST_MEAN).div(MNIST_STD).unsqueeze(1)
        self.by_digit = [images[labels == digit] for digit in range(10)]       # (count, 1, 28, 28) each

    def show(self, digits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """An image of every digit, picked at random from that digit's images.     (B, n) -> (B, n, 1, 28, 28)"""
        shown = torch.empty(*digits.shape, 1, 28, 28)
        for digit, pool in enumerate(self.by_digit):
            where = digits == digit
            picks = torch.randint(0, len(pool), (int(where.sum()),), generator=generator)
            shown[where] = pool[picks]
        return shown


def sample_episodes(batch_size: int,
                    generator: torch.Generator,
                    n_digits: int,
                    n_ops: int,
                    ticks_per_item: int = 3,
                    n_classes: int = 10,
                    images: MNISTImages | None = None):
    """
    Returns:
        x       (B, n_digits * R, 10, 1, 1)     one-hot digits, each repeated for the R ticks it is shown
                (B, n_digits * R, 1, 28, 28)    or, given `images`, an MNIST image of every digit
        z       (B, (1 + 2 n_ops) * R)          question tokens, each repeated for R ticks
        answer  (B,)                            the right answer, 0 to 9

    The images are picked after the digits, the questions and the operators, so the digits, questions and
    answers are the same with or without images for the first batch drawn from a generator.
    """
    digits = torch.randint(0, n_classes, (batch_size, n_digits), generator=generator)
    asked_positions = torch.randint(0, n_digits, (batch_size, n_ops + 1), generator=generator)
    use_minus = torch.randint(0, 2, (batch_size, n_ops), generator=generator).bool()

    if images is None:
        # The digits, as one-hot vectors
        x = torch.zeros(batch_size, n_digits, n_classes, 1, 1)
        x.scatter_(2, digits[:, :, None, None, None], 1.0)
    else:
        x = images.show(digits, generator)

    # Spell out the question one part at a time, and work out the answer along the way
    episode = torch.arange(batch_size)
    answer = digits[episode, asked_positions[:, 0]]
    question_parts = [asked_positions[:, 0]]

    for op in range(n_ops):
        next_digit = digits[episode, asked_positions[:, op + 1]]
        added = (answer + next_digit) % 10
        subtracted = (answer - next_digit) % 10
        answer = torch.where(use_minus[:, op], subtracted, added)

        operator_token = torch.where(use_minus[:, op], MINUS, PLUS)
        question_parts.append(operator_token)
        question_parts.append(asked_positions[:, op + 1])

    z = torch.stack(question_parts, dim=1)

    # Every item is shown for R ticks
    x = x.repeat_interleave(ticks_per_item, dim=1)
    z = z.repeat_interleave(ticks_per_item, dim=1)
    return x, z, answer
