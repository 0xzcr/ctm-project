"""
Train and evaluate the CTM with the diary, or without it for comparison, on symbolic Q&A-MNIST.

    python train.py                          the CTM with the diary, seed 0
    python train.py --no-diary               the plain CTM: same seed, same starting weights, same data
    python train.py --seed 3 --device cuda
    python train.py --no-diary --d-model 107 the plain CTM with as many parameters as the CTM with the diary
    python train.py --latest-head            the diary with a fifth head that reads the latest page
    python train.py --images                 real MNIST images instead of one-hot digits
    python train.py --images --pretrain-backbone --steps 16000
                                             the image backbone first learns to tell digits apart, then is frozen

Training follows our earlier experiments (thesis Box 1): 1-4 digits and 0-3 operations, 4000 steps of 64
episodes, AdamW at 3e-3 with 200 warm-up steps and a cosine decay, gradients clipped at 1. The loss and
the read-out use the upstream rule: the most certain of the answer ticks.

Evaluation reports accuracy by the number of digits, averaged over 0-3 operations: 1-4 digits as in
training, and 5, 6, 8, 12 and 16 digits for length (all well within the clock's horizon of 50 pages).
With --images, training shows MNIST's training images and evaluation shows its test images.
The results and the trained weights are saved in results/.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cem.model import build_model
from cem.task import MNISTImages, sample_episodes
from utils.losses import qamnist_loss  # noqa: E402  (upstream; importing cem put it on the path)

TICKS_PER_ITEM = 3                     # R: every digit, every question part and the answer last 3 ticks
TRAIN_DIGITS = (1, 2, 3, 4)
TRAIN_OPS = (0, 1, 2, 3)
LONGER_DIGITS = (5, 6, 8, 12, 16)

WARMUP_STEPS = 200
GRADIENT_CLIP = 1.0
EVAL_EPISODES = 1000                   # for every (digits, operations) pair

# --pretrain-backbone: the image backbone is first trained as a digit classifier, on MNIST's training images
PRETRAIN_STEPS = 1000
PRETRAIN_BATCH = 128
PRETRAIN_LR = 3e-3                     # 0.94 of MNIST's test digits after 1000 steps (1e-3 reaches 0.89)
PRETRAIN_TEST_IMAGES = 2000

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-diary", action="store_true", help="train the plain CTM instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3, help="the peak learning rate")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--d-model", type=int, default=64, help="the number of CTM neurons")
    parser.add_argument("--latest-head", action="store_true", help="give the diary a head that reads the latest page")
    parser.add_argument("--images", action="store_true", help="show the digits as real MNIST images")
    parser.add_argument("--pretrain-backbone", action="store_true",
                        help="with --images: first train the image backbone to tell the digits apart, then freeze it")
    parser.add_argument("--results", default=RESULTS_FOLDER, help="the folder to save the results in")
    args = parser.parse_args()
    if args.latest_head and args.no_diary:
        parser.error("--latest-head needs the diary")
    if args.pretrain_backbone and not args.images:
        parser.error("--pretrain-backbone needs --images")
    return args


def variant_name(args) -> str:
    """ctm or ctm_with_diary, then _latest, _d<neurons> (if not 64), _mnist, _pretrained and _steps<N> (if not
    4000) for the options used."""
    if args.no_diary:
        name = "ctm"
    else:
        name = "ctm_with_diary"
    if args.latest_head:
        name += "_latest"
    if args.d_model != 64:
        name += f"_d{args.d_model}"
    if args.images:
        name += "_mnist"
    if args.pretrain_backbone:
        name += "_pretrained"
    if args.steps != 4000:
        name += f"_steps{args.steps}"
    return name


def learning_rate_factor(step: int, total_steps: int) -> float:
    """The share of the peak learning rate at this step: a linear warm-up, then a cosine decay to zero."""
    warm = min(1.0, (step + 1) / WARMUP_STEPS)
    steps_into_decay = max(0, step - WARMUP_STEPS)
    decay_steps = max(1, total_steps - WARMUP_STEPS)
    # pi * a / b, in this order, rounds exactly as our earlier runs did, so the plain CTM reproduces them bit for bit
    cosine = 0.5 * (1.0 + math.cos(math.pi * steps_into_decay / decay_steps))
    return warm * cosine


def answer_ticks(predictions: torch.Tensor, certainties: torch.Tensor):
    """Keep only the R answer ticks at the end of every episode."""
    return predictions[..., -TICKS_PER_ITEM:], certainties[..., -TICKS_PER_ITEM:]


# -------------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------------

def pretrain_backbone(model, train_images, test_images, seed: int, device) -> float:
    """
    Teach the image backbone to tell the digits apart before the Q&A training: a digit classifier made of the
    backbone, the average of its 7x7 features and a linear read-out, trained on MNIST's training images. In 4000
    steps of Q&A training from scratch, most seeds never learn to name even a single digit, so without this the
    image runs measure learning to see rather than remembering. The backbone is then frozen: trained on along with
    the CTM at the Q&A learning rate, it lost a fifth of what it had learned and the CTM did no better.
    Returns the classifier's accuracy on PRETRAIN_TEST_IMAGES test images.
    """
    backbone = model.backbone_digit
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2000 + seed)                           # the same read-out for both variants of a seed
        read_out = nn.Linear(model.d_input, 10).to(device)
    optimiser = torch.optim.AdamW(list(backbone.parameters()) + list(read_out.parameters()), lr=PRETRAIN_LR,
                                  weight_decay=0.0)
    generator = torch.Generator().manual_seed(2000 + seed)       # its own data, so the Q&A data stay the same

    for _ in range(PRETRAIN_STEPS):
        digits = torch.randint(0, 10, (PRETRAIN_BATCH, 1), generator=generator)
        shown = train_images.show(digits, generator)[:, 0]                         # (B, 1, 28, 28)
        logits = read_out(backbone(shown.to(device)).mean(dim=(2, 3)))             # (B, 10)
        loss = F.cross_entropy(logits, digits[:, 0].to(device))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    backbone.eval()
    with torch.no_grad():
        digits = torch.randint(0, 10, (PRETRAIN_TEST_IMAGES, 1), generator=generator)
        shown = test_images.show(digits, generator)[:, 0]
        guesses = read_out(backbone(shown.to(device)).mean(dim=(2, 3))).argmax(dim=-1)
    model.freeze_backbone()
    return (guesses == digits[:, 0].to(device)).float().mean().item()


def train(model, args, device, images=None) -> list[float]:
    """Train the model in place. Returns the mean loss of every 100 steps. `images`: MNISTImages, or None."""
    learning = [parameter for parameter in model.parameters() if parameter.requires_grad]   # not a frozen backbone
    optimiser = torch.optim.AdamW(learning, lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: learning_rate_factor(step, args.steps))

    # The data depend only on the seed, so the two variants of a seed train on the same episodes.
    data_generator = torch.Generator().manual_seed(1000 + args.seed)
    size_picker = np.random.default_rng(args.seed)

    model.train()
    loss_curve = []
    recent_losses = []
    for step in range(args.steps):
        # Every batch has one task size, drawn uniformly from the training range.
        n_digits = int(size_picker.integers(TRAIN_DIGITS[0], TRAIN_DIGITS[-1] + 1))
        n_ops = int(size_picker.integers(TRAIN_OPS[0], TRAIN_OPS[-1] + 1))
        x, z, answer = sample_episodes(args.batch, data_generator, n_digits, n_ops, TICKS_PER_ITEM, images=images)

        predictions, certainties, _ = model(x.to(device), z.to(device))
        predictions, certainties = answer_ticks(predictions, certainties)
        loss, _ = qamnist_loss(predictions, certainties, answer.to(device), use_most_certain=True)

        if not torch.isfinite(loss):
            print(f"step {step}: the loss is not finite, so training stops here")
            break

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimiser.step()
        schedule.step()

        recent_losses.append(loss.item())
        if (step + 1) % 100 == 0:
            mean_loss = float(np.mean(recent_losses))
            loss_curve.append(mean_loss)
            recent_losses = []
            print(f"step {step + 1:5d}    loss {mean_loss:.4f}    learning rate {schedule.get_last_lr()[0]:.2e}")

    return loss_curve


# -------------------------------------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model, n_digits: int, n_ops: int, device, images=None) -> float:
    """
    The share of EVAL_EPISODES episodes answered right, read out at the most certain answer tick.
    The episodes depend only on the task size, so every model and every seed is tested on the same ones.
    They go through in one batch, as in our earlier runs: splitting them changes the last digits of the outputs.
    """
    generator = torch.Generator().manual_seed(10_000 + 1000 * n_digits + n_ops)
    x, z, answer = sample_episodes(EVAL_EPISODES, generator, n_digits, n_ops, TICKS_PER_ITEM, images=images)

    predictions, certainties, _ = model(x.to(device), z.to(device))
    predictions, certainties = answer_ticks(predictions, certainties)

    most_certain_tick = certainties[:, 1].argmax(dim=-1)                           # (B,)
    episode = torch.arange(EVAL_EPISODES, device=device)
    guess = predictions[episode, :, most_certain_tick].argmax(dim=-1)              # (B,)

    n_right = int((guess == answer.to(device)).sum())
    return n_right / EVAL_EPISODES


def accuracy_for_every_ops(model, n_digits: int, device, images=None) -> dict[int, float]:
    """{number of operations: accuracy}, for 0-3 operations."""
    by_ops = {}
    for n_ops in TRAIN_OPS:
        by_ops[n_ops] = accuracy(model, n_digits, n_ops, device, images)
    return by_ops


def evaluate(model, device, images=None) -> dict[int, dict]:
    """
    Accuracy with every number of digits, averaged over 0-3 operations as in training.
    Returns {n_digits: {"mean": accuracy, "by_ops": {n_ops: accuracy}}}.
    """
    model.eval()
    report = {}
    for n_digits in TRAIN_DIGITS + LONGER_DIGITS:
        by_ops = accuracy_for_every_ops(model, n_digits, device, images)
        mean = sum(by_ops.values()) / len(by_ops)
        report[n_digits] = {"mean": mean, "by_ops": by_ops}
    model.train()
    return report


def print_report(report: dict, variant: str):
    print(f"\naccuracy of {variant} by number of digits (mean over 0-3 operations)")
    for n_digits, row in report.items():
        if n_digits in TRAIN_DIGITS:
            note = "as in training"
        else:
            note = "longer than in training"
        print(f"    {n_digits:2d} digits    {row['mean']:.3f}    {note}")


# -------------------------------------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # One CPU thread, as in our earlier runs. The thread count changes how sums are rounded, and with one thread
    # the plain CTM reproduces those runs exactly.
    torch.set_num_threads(1)

    device = torch.device(args.device)
    variant = variant_name(args)

    train_images = None
    test_images = None
    if args.images:
        train_images = MNISTImages("train")
        test_images = MNISTImages("test")          # evaluation shows images the model never saw in training

    model = build_model(use_diary=not args.no_diary, seed=args.seed, ticks_per_item=TICKS_PER_ITEM,
                        d_model=args.d_model, latest_head=args.latest_head, images=args.images)
    model = model.to(device)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{variant}, seed {args.seed}: {n_parameters} parameters")

    started = time.time()
    backbone_accuracy = None
    if args.pretrain_backbone:
        backbone_accuracy = pretrain_backbone(model, train_images, test_images, args.seed, device)
        print(f"the pretrained backbone names {backbone_accuracy:.3f} of the test digits it is shown")

    loss_curve = train(model, args, device, train_images)
    report = evaluate(model, device, test_images)
    minutes = (time.time() - started) / 60
    print_report(report, variant)

    os.makedirs(args.results, exist_ok=True)
    name = os.path.join(args.results, f"{variant}_seed{args.seed}")
    torch.save(model.state_dict(), name + ".pt")

    summary = {
        "variant": variant,
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "peak_learning_rate": args.lr,
        "device": args.device,
        "d_model": args.d_model,
        "latest_head": args.latest_head,
        "images": args.images,
        "pretrained_backbone_test_accuracy": backbone_accuracy,
        "parameters": n_parameters,
        "minutes": round(minutes, 1),
        "accuracy": report,
        "loss_every_100_steps": loss_curve,
    }
    with open(name + ".json", "w") as file:
        json.dump(summary, file, indent=2)
    print(f"\nsaved {name}.json and {name}.pt after {minutes:.1f} minutes")


if __name__ == "__main__":
    main()
