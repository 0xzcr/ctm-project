"""
Tests that the code does what the thesis says it does. Every test names the result it checks.

    python -m pytest test_cem.py

The clock, the sketch and the diary are tested on their own first, then the whole model, then the data.
"""
import hashlib
import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cem.clock import Clock
from cem.memory import FOCUS, VALUE_SIZE, ClockedEventMemory, event_sync
from cem.model import build_model
from cem.sketch import SparseSketch
from cem.task import MNIST_MEAN, MNIST_STD, PLUS, MNISTImages, sample_episodes
from models.ctm_qamnist import ContinuousThoughtMachineQAMNIST  # noqa: E402  (upstream; importing cem put it on the path)
from utils.losses import qamnist_loss  # noqa: E402  (upstream)

import train
from verify import ABLATION_DIGITS, lost_accuracy, mass_by_gap, neighbour_share, silence, slip_chances

# The sizes build_model() gives the diary
N_PAIRS = 136              # 16 synchronised neurons -> 16 * 17 / 2 pairs
D_MODEL = 64
D_INPUT = 32
TICKS_PER_ITEM = 3

MARGIN = Clock().margin    # m_H of the design point


# -------------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------------

def diary_with_pages(n_pages: int, batch_size: int, dtype: torch.dtype = torch.float32,
                     latest_head: bool = False) -> ClockedEventMemory:
    """A diary in which each of `batch_size` episodes has written `n_pages` random pages."""
    diary = ClockedEventMemory(N_PAIRS, D_MODEL, D_INPUT, seed=0, latest_head=latest_head).to(dtype)
    diary.reset(batch_size, torch.device("cpu"), dtype)

    generator = torch.Generator().manual_seed(0)
    for _ in range(n_pages):
        event = torch.randn(batch_size, N_PAIRS, generator=generator, dtype=dtype)
        seen = torch.randn(batch_size, D_INPUT, generator=generator, dtype=dtype)
        diary.write(event, seen)
    return diary


def fake_mnist(folder) -> MNISTImages:
    """A tiny stand-in for Keras's MNIST file, in which every pixel of an image is its digit (3 images per digit)."""
    labels = np.arange(30, dtype=np.uint8) % 10
    pixels = np.repeat(labels, 28 * 28).reshape(30, 28, 28)
    path = folder / "mnist.npz"
    np.savez(path, x_train=pixels, y_train=labels, x_test=pixels, y_test=labels)
    return MNISTImages("train", str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def lowest_light_on_the_named_page(n_pages: int) -> float:
    """Episode i asks the by-number head for page i. Returns the smallest light any episode put on its page."""
    diary = diary_with_pages(n_pages, batch_size=n_pages)
    every_page = torch.arange(n_pages)

    diary.open_at(every_page)
    _, light = diary.read_by_number(diary.clock.stamp(every_page))     # light[i, j]: episode i, page j

    light_on_the_named_page = light.diagonal()
    return light_on_the_named_page.min().item()


def lowest_grip_while_flipping(n_pages: int, direction: int) -> float:
    """
    Episode i opens the diary at page i, then flips one page per read (direction +1: the next-page head,
    -1: the previous-page head) until it runs off the end of the diary. Returns the smallest light any
    head put on the page it should be on. Also checks that a head past the end is silent.
    """
    diary = diary_with_pages(n_pages, batch_size=n_pages)
    start_pages = torch.arange(n_pages)
    page_stamps = diary.clock.stamp(start_pages)
    episodes = torch.arange(n_pages)

    diary.open_at(start_pages)

    lowest_grip = 1.0
    for flips in range(1, n_pages):
        if direction > 0:
            _, light = diary.read_next(page_stamps)
        else:
            _, light = diary.read_previous(page_stamps)

        right_page = start_pages + direction * flips                # where each head should be now: (B,)
        exists = (right_page >= 0) & (right_page < n_pages)

        assert torch.all(light[~exists] == 0), "a head that ran off the diary must read nothing"

        grip = light[episodes[exists], right_page[exists]]
        lowest_grip = min(lowest_grip, grip.min().item())
    return lowest_grip


def largest_spill_of_one_edit(n_pages: int) -> tuple[float, float]:
    """
    Episode i opens page i with the by-number head and crosses it off with a random edit e.
    Returns (the largest move of any other page, as a share of ||e||;  Theorem 14(b)'s bound 2 (1 + n)^(-c m_H)).
    Runs in float64: at 50 pages the bound is 2.3e-7, below float32's rounding of a page of length 8.
    """
    diary = diary_with_pages(n_pages, batch_size=n_pages, dtype=torch.float64)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        random_direction = torch.randn(diary.edit_direction.weight.shape, generator=generator, dtype=torch.float64)
        diary.edit_direction.weight.copy_(random_direction)          # W_e != 0, so the edit is not empty

    every_page = torch.arange(n_pages)
    diary.open_at(every_page)
    _, light = diary.read_by_number(diary.clock.stamp(every_page))
    diary.edit_light = light                                          # edit through the by-number head only

    neurons = torch.randn(n_pages, D_MODEL, generator=generator, dtype=torch.float64)
    edit = torch.sigmoid(diary.edit_strength(neurons)) * diary.edit_direction(neurons)      # e: (B, d_v)

    values_before = diary.values.clone()
    sketches_before = diary.sketches.clone()
    with torch.no_grad():
        diary.cross_off(neurons)

    assert torch.equal(diary.sketches, sketches_before), "edits must never change a sketch (Theorem 14(a))"

    moved = (diary.values - values_before).norm(dim=-1)              # how far page j moved in episode i: (B, n)
    edit_size = edit.norm(dim=-1, keepdim=True)                       # (B, 1)
    share_moved = moved / edit_size                                   # (B, n)

    other_pages = ~torch.eye(n_pages, dtype=torch.bool)
    largest_spill = share_moved[other_pages].max().item()
    bound = 2 * (1 + n_pages) ** (-FOCUS * diary.clock.margin)
    return largest_spill, bound


def gradient_size(parameter: torch.nn.Parameter) -> float:
    """The length of a parameter's gradient, 0 if it got none."""
    if parameter.grad is None:
        return 0.0
    return parameter.grad.norm().item()


def diary_gradient_sizes(model, x, z, answer) -> dict[str, float]:
    """One forward and backward pass with the upstream loss. Returns the size of the gradient on W_o, W_v and W_e."""
    model.zero_grad()
    predictions, certainties, _ = model(x, z)
    answer_predictions = predictions[..., -TICKS_PER_ITEM:]
    answer_certainties = certainties[..., -TICKS_PER_ITEM:]
    loss, _ = qamnist_loss(answer_predictions, answer_certainties, answer, use_most_certain=True)
    loss.backward()

    return {
        "read_out": gradient_size(model.diary.output.weight),             # W_o
        "write": gradient_size(model.diary.write_value.weight),           # W_v
        "edit": gradient_size(model.diary.edit_direction.weight),         # W_e
    }


def reference_answer(digits: list[int], tokens: list[int]) -> int:
    """The answer to one question, worked out in plain Python: index, operator, index, operator, ..., index."""
    result = digits[tokens[0]]
    for position in range(1, len(tokens), 2):
        operator = tokens[position]
        next_digit = digits[tokens[position + 1]]
        if operator == PLUS:
            result = result + next_digit
        else:
            result = result - next_digit
    return result % 10


# -------------------------------------------------------------------------------------------------------
# The clock (cem/clock.py)
# -------------------------------------------------------------------------------------------------------

def test_the_clock_has_the_designed_margin():
    """Eq. 5: 16 hands and a horizon of 49 pages give trust weights with margin m_H = 0.325."""
    clock = Clock(n_hands=16, horizon=49)

    assert abs(clock.margin - 0.325) < 0.0005
    assert torch.all(clock.trust >= 0)
    assert abs(clock.trust.sum().item() - 1.0) < 1e-6


def test_agreement_depends_only_on_the_gap():
    """
    Eq. 4 and Theorem 11: agreement(Phi(i), Phi(j)) = kappa(i - j). A page agrees with itself exactly 1, and
    with every other page within the horizon at most 1 - m_H.
    """
    clock = Clock()
    pages = torch.arange(50)
    stamps = clock.stamp(pages)                                       # (50, 2K)
    agreement = clock.agreement(stamps, stamps)                       # (50, 50): row i asks for page i

    # kappa(i - j) = sum_k w_k cos(omega_k (i - j)), straight from its definition
    gaps = pages.unsqueeze(1) - pages.unsqueeze(0)                    # gaps[i, j] = i - j
    angles = gaps.unsqueeze(-1) * clock.speeds                        # (50, 50, K)
    kappa = (clock.trust * torch.cos(angles)).sum(dim=-1)             # (50, 50)
    assert torch.allclose(agreement, kappa, atol=1e-5)

    same_page = torch.eye(50, dtype=torch.bool)
    assert torch.allclose(agreement[same_page], torch.ones(50), atol=1e-5)
    assert agreement[~same_page].max().item() <= 1 - clock.margin + 1e-5


def test_turning_the_hands_gives_the_stamp_of_another_page():
    """Theorem 11: turn(Phi(j), delta) = Phi(j + delta), for every page j and every delta."""
    clock = Clock()
    pages = torch.arange(50)

    for delta in (1, -1, 7, -20):
        turned = clock.turn(clock.stamp(pages), delta)
        assert torch.allclose(turned, clock.stamp(pages + delta), atol=1e-5)


# -------------------------------------------------------------------------------------------------------
# The sketch (cem/sketch.py)
# -------------------------------------------------------------------------------------------------------

def test_a_sketch_has_a_fixed_number_of_units_on():
    """Eq. 8: exactly f*M = 102 units are on, and every sketch has length 1, so overlaps lie in [0, 1]."""
    sketch = SparseSketch(D_INPUT, n_units=2048, on_fraction=0.05, seed=0)
    features = torch.randn(100, D_INPUT, generator=torch.Generator().manual_seed(0))

    sketches = sketch(features)
    units_on = (sketches > 0).sum(dim=-1)

    assert torch.all(units_on == 102)
    assert torch.allclose(sketches.norm(dim=-1), torch.ones(100))


def test_sparse_sketches_separate_similar_inputs():
    """Theorem 12(a, b): at f = 5%, the sketches of two inputs with cosine 0.9 share only about 64% of their units."""
    sketch = SparseSketch(D_INPUT, seed=0)
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(2000, D_INPUT, generator=generator)
    noise = torch.randn(2000, D_INPUT, generator=generator)

    # `second` has cosine exactly 0.9 with `first`: 0.9 of first's direction, plus sqrt(1 - 0.81) of a
    # direction at right angles to it.
    along = F.normalize(first, dim=-1)
    across = noise - (noise * along).sum(dim=-1, keepdim=True) * along
    across = F.normalize(across, dim=-1)
    second = 0.9 * along + math.sqrt(1 - 0.9 ** 2) * across

    overlap = (sketch(first) * sketch(second)).sum(dim=-1)            # the share of on-units in common: (2000,)
    assert abs(overlap.mean().item() - 0.64) < 0.02


# -------------------------------------------------------------------------------------------------------
# The diary (cem/memory.py)
# -------------------------------------------------------------------------------------------------------

def test_a_page_holds_exactly_its_event():
    """
    Theorem 15 (eq. 2): subtracting the faded tallies from just before an event leaves exactly the event's
    own ticks. Checked against the upstream tallies, with a random fading factor for every pair.
    """
    model = build_model(use_diary=False, seed=0)
    generator = torch.Generator().manual_seed(0)
    n_ticks = 10
    neuron_outputs = torch.randn(n_ticks, 4, D_MODEL, generator=generator)        # z_t for every tick
    fade = torch.rand(4, N_PAIRS, generator=generator)                              # r, anywhere in (0, 1)

    # The upstream tallies after every tick
    alphas = []
    betas = []
    alpha, beta = None, None
    for tick in range(n_ticks):
        _, alpha, beta = model.compute_synchronisation(neuron_outputs[tick], alpha, beta, fade, synch_type="out")
        alphas.append(alpha)
        betas.append(beta)

    # An event from tick s = 4 to tick f = 6, read off the tallies
    first_tick = 4
    last_tick = 6
    n_event_ticks = last_tick - first_tick + 1
    from_tallies = event_sync(alphas[last_tick], betas[last_tick],
                              alphas[first_tick - 1], betas[first_tick - 1],
                              fade, n_event_ticks)

    # The same event from scratch: sum of r^(f - t) z_i z_j over its ticks, over the root of sum of r^(f - t).
    # A fresh upstream call (no tallies yet) returns this tick's products z_i z_j as its alpha.
    weighted_products = torch.zeros_like(fade)
    total_weight = torch.zeros_like(fade)
    for tick in range(first_tick, last_tick + 1):
        _, products, _ = model.compute_synchronisation(neuron_outputs[tick], None, None, fade, synch_type="out")
        weight = fade ** (last_tick - tick)
        weighted_products = weighted_products + weight * products
        total_weight = total_weight + weight
    from_scratch = weighted_products / torch.sqrt(total_weight)

    assert torch.allclose(from_tallies, from_scratch, atol=1e-5)


def test_writing_adds_one_page_of_fixed_length():
    """
    Eq. 2: every write adds one page, and every page has length sqrt(d_v) = 8 to float precision (LN has no
    learned scale). With PyTorch's default eps the pages would be about 3e-4 short, and this test would fail.
    """
    diary = diary_with_pages(n_pages=5, batch_size=3)

    assert diary.n_pages == 5
    page_lengths = diary.values.norm(dim=-1)                          # (3, 5)
    assert torch.allclose(page_lengths, torch.full_like(page_lengths, math.sqrt(VALUE_SIZE)), atol=1e-5)


def test_reads_stay_bounded_however_many_pages():
    """Theorem 3(b): ||r_t|| <= ||W_o||_2 sqrt(4 d_v) + ||b_o||, for any number of pages, even far past 50."""
    diary = diary_with_pages(n_pages=200, batch_size=8)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        diary.output.weight.copy_(torch.randn(diary.output.weight.shape, generator=generator))
        diary.output.bias.copy_(torch.randn(diary.output.bias.shape, generator=generator))

    diary.open_at(torch.tensor([0, 3, 10, 50, 77, 100, 150, 199]))
    seen = torch.randn(8, D_INPUT, generator=generator)
    with torch.no_grad():
        read = diary.read(seen)

    weight_norm = torch.linalg.matrix_norm(diary.output.weight, ord=2)
    bound = weight_norm * math.sqrt(4 * VALUE_SIZE) + diary.output.bias.norm()
    assert torch.all(read.norm(dim=-1) <= bound)


def test_the_by_number_head_opens_the_right_page():
    """
    Theorem 5: asked for page i among n pages, the right page gets at least 1 / (1 + (n - 1)(1 + n)^(-c m_H))
    of the light. Over 2 to 50 pages that guarantee is never below 0.9887 (its lowest point, at n = 2).
    """
    lowest_guarantee = 1.0
    for n_pages in range(2, 51):
        guarantee = 1 / (1 + (n_pages - 1) * (1 + n_pages) ** (-FOCUS * MARGIN))
        assert lowest_light_on_the_named_page(n_pages) >= guarantee
        lowest_guarantee = min(lowest_guarantee, guarantee)

    assert round(lowest_guarantee, 4) == 0.9887


def test_flipping_pages_keeps_its_grip():
    """
    Theorem 9: starting from an exact stamp, every flip puts at least 0.95 of the light on the right page, for
    as long as pages exist, in either direction and in every diary of 2 to 50 pages.
    """
    for n_pages in range(2, 51):
        assert lowest_grip_while_flipping(n_pages, direction=+1) >= 0.95
        assert lowest_grip_while_flipping(n_pages, direction=-1) >= 0.95


def test_the_content_head_finds_the_page_that_looked_the_same():
    """Theorem 12(c): a cue that looks exactly like one event finds that event's page among 50."""
    n_pages = 50
    diary = ClockedEventMemory(N_PAIRS, D_MODEL, D_INPUT, seed=0)
    diary.reset(batch_size=n_pages, device=torch.device("cpu"), dtype=torch.float32)

    # Every episode writes the same 50 pages. Page p's event looked like looks[p].
    generator = torch.Generator().manual_seed(0)
    looks = torch.randn(n_pages, D_INPUT, generator=generator)
    for page in range(n_pages):
        event = torch.randn(n_pages, N_PAIRS, generator=generator)
        seen = looks[page].expand(n_pages, D_INPUT)
        diary.write(event, seen)

    # Episode i is cued with what page i looked like.
    _, light = diary.read_by_content(looks)                           # (50, 50)
    assert light.diagonal().min().item() >= 0.99


def test_an_edit_stays_on_its_page():
    """
    Theorem 14: an edit e made through the by-number head never changes a sketch, and moves every other page by
    at most 2 (1 + n)^(-c m_H) ||e||: 2.3% of the edit at n = 2 pages, far less with more pages.
    """
    for n_pages in range(2, 51):
        largest_spill, bound = largest_spill_of_one_edit(n_pages)
        assert largest_spill <= bound

    _, bound_at_two_pages = largest_spill_of_one_edit(2)
    assert round(bound_at_two_pages, 3) == 0.023


def test_the_latest_page_head_reads_the_last_page_exactly():
    """
    The optional fifth head (not in the thesis) reads the page of the event that has just ended, exactly, as a
    fifth block of the read. It starts silent like the other four: W_o = b_o = 0.
    """
    diary = diary_with_pages(n_pages=4, batch_size=3, latest_head=True)
    seen = torch.randn(3, D_INPUT)

    read, light = diary.read_latest()
    assert torch.equal(read, diary.values[:, -1])
    assert light.tolist() == [[0.0, 0.0, 0.0, 1.0]] * 3
    assert torch.equal(diary.read(seen), torch.zeros(3, D_INPUT))

    # Let W_o pass the first D_INPUT numbers of the fifth block through: the read is then the latest page.
    with torch.no_grad():
        diary.output.weight[:, 4 * VALUE_SIZE:4 * VALUE_SIZE + D_INPUT] = torch.eye(D_INPUT)
    assert torch.allclose(diary.read(seen), diary.values[:, -1, :D_INPUT])


# -------------------------------------------------------------------------------------------------------
# The whole model (cem/model.py)
# -------------------------------------------------------------------------------------------------------

def test_without_the_diary_the_model_is_the_upstream_model():
    """cem/model.py writes out the upstream forward pass with the diary's steps added. Without the diary it must
    compute exactly what the upstream forward pass computes."""
    model = build_model(use_diary=False, seed=0)
    x, z, _ = sample_episodes(8, torch.Generator().manual_seed(0), n_digits=4, n_ops=3)

    with torch.no_grad():
        ours = model(x, z)
        upstream = ContinuousThoughtMachineQAMNIST.forward(model, x, z)

    for our_output, upstream_output in zip(ours, upstream):
        assert torch.equal(our_output, upstream_output)


def test_the_diary_changes_nothing_at_the_start():
    """
    Theorem 3(a): with W_o = b_o = W_e = 0, the CTM with the diary computes exactly what the CTM without it
    computes. With the same seed both also start from exactly the same CTM weights (a paired comparison).
    """
    plain = build_model(use_diary=False, seed=0)
    with_diary = build_model(use_diary=True, seed=0)

    plain_weights = plain.state_dict()
    diary_weights = with_diary.state_dict()
    for name, weight in plain_weights.items():
        assert torch.equal(weight, diary_weights[name]), name

    extra_names = set(diary_weights) - set(plain_weights)
    for name in extra_names:
        assert name.startswith("diary."), name

    x, z, _ = sample_episodes(8, torch.Generator().manual_seed(0), n_digits=4, n_ops=3)
    with torch.no_grad():
        plain_output = plain(x, z)
        diary_output = with_diary(x, z)

    for plain_part, diary_part in zip(plain_output, diary_output):
        assert torch.equal(plain_part, diary_part)


def test_with_images_the_diary_still_changes_nothing_at_the_start():
    """Theorem 3(a) with real images: the upstream image backbone stays, and starts the same with or without the diary."""
    plain = build_model(use_diary=False, seed=0, images=True)
    with_diary = build_model(use_diary=True, seed=0, images=True)
    assert type(plain.backbone_digit).__name__ == "MNISTBackbone"

    images = torch.randn(4, 3, 1, 28, 28).repeat_interleave(TICKS_PER_ITEM, dim=1)     # three digits
    _, z, _ = sample_episodes(4, torch.Generator().manual_seed(0), n_digits=3, n_ops=2)
    with torch.no_grad():
        plain_output = plain(images, z)
        diary_output = with_diary(images, z)

    for plain_part, diary_part in zip(plain_output, diary_output):
        assert torch.equal(plain_part, diary_part)


def test_both_variants_start_from_the_same_pretrained_backbone(tmp_path, monkeypatch):
    """
    --pretrain-backbone keeps the comparison paired: with the same seed, both variants get the same backbone. It is
    frozen afterwards: training mode leaves its batch norm in eval mode, and an episode changes nothing in it.
    """
    images = fake_mnist(tmp_path)
    monkeypatch.setattr(train, "PRETRAIN_STEPS", 3)
    monkeypatch.setattr(train, "PRETRAIN_TEST_IMAGES", 20)

    backbones = []
    for use_diary in (False, True):
        model = build_model(use_diary=use_diary, seed=0, images=True)
        train.pretrain_backbone(model, images, images, seed=0, device="cpu")
        backbones.append({name: weight.clone() for name, weight in model.backbone_digit.state_dict().items()})

    for name, weight in backbones[0].items():
        assert torch.equal(weight, backbones[1][name]), name

    model.train()
    assert not model.backbone_digit.training
    assert not any(parameter.requires_grad for parameter in model.backbone_digit.parameters())
    x, z, _ = sample_episodes(4, torch.Generator().manual_seed(0), n_digits=2, n_ops=1, images=images)
    model(x, z)
    for name, weight in model.backbone_digit.state_dict().items():
        assert torch.equal(weight, backbones[1][name]), name


def test_learning_starts_in_order():
    """
    Proposition 4: at the start only the read-out W_o gets a gradient; W_v and W_e get exactly zero. From the
    first update that makes W_o non-zero, W_v and W_e get gradients too.
    """
    model = build_model(use_diary=True, seed=0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    x, z, answer = sample_episodes(16, torch.Generator().manual_seed(0), n_digits=3, n_ops=2)

    at_the_start = diary_gradient_sizes(model, x, z, answer)
    assert at_the_start["read_out"] > 0
    assert at_the_start["write"] == 0
    assert at_the_start["edit"] == 0

    optimiser.step()                                                  # W_o is no longer zero
    assert model.diary.output.weight.abs().sum().item() > 0

    after_one_update = diary_gradient_sizes(model, x, z, answer)
    assert after_one_update["write"] > 0
    assert after_one_update["edit"] > 0


def test_every_event_writes_one_page():
    """Assumption A1: every digit and every part of the question writes one page. The answer ticks write none."""
    model = build_model(use_diary=True, seed=0)
    n_digits = 5
    n_ops = 2
    x, z, _ = sample_episodes(2, torch.Generator().manual_seed(0), n_digits, n_ops)

    with torch.no_grad():
        model(x, z)

    n_question_parts = 1 + 2 * n_ops
    assert model.diary.n_pages == n_digits + n_question_parts


# -------------------------------------------------------------------------------------------------------
# The data (cem/task.py)
# -------------------------------------------------------------------------------------------------------

def test_the_answers_are_right():
    """Section 2.2: the answer is the question's running result, modulo 10. Checked against plain Python."""
    assert reference_answer([7, 2, 9], [2, PLUS, 0]) == 6            # the example in cem/task.py

    n_episodes = 200
    x, z, answer = sample_episodes(n_episodes, torch.Generator().manual_seed(0), n_digits=6, n_ops=3)

    # Every item is shown for R ticks, so keep one tick of each.
    digits = x[:, ::TICKS_PER_ITEM].argmax(dim=2).flatten(1)          # (B, 6)
    tokens = z[:, ::TICKS_PER_ITEM]                                    # (B, 7)

    for episode in range(n_episodes):
        expected = reference_answer(digits[episode].tolist(), tokens[episode].tolist())
        assert answer[episode].item() == expected


def test_every_digit_is_shown_as_one_of_its_own_images(tmp_path):
    """
    With images, each digit is shown as an image of that digit, and the digits, questions and answers are the
    ones the one-hot episodes have.
    """
    images = fake_mnist(tmp_path)

    x_one_hot, z_one_hot, answer_one_hot = sample_episodes(50, torch.Generator().manual_seed(0), 5, 2)
    x, z, answer = sample_episodes(50, torch.Generator().manual_seed(0), 5, 2, images=images)

    assert x.shape == (50, 5 * TICKS_PER_ITEM, 1, 28, 28)
    digit_in_image = (x[:, :, 0, 0, 0] * MNIST_STD + MNIST_MEAN) * 255             # undo the normalisation
    digit_one_hot = x_one_hot.argmax(dim=2).flatten(1).float()
    assert torch.allclose(digit_in_image, digit_one_hot, atol=1e-3)
    assert torch.equal(z, z_one_hot)
    assert torch.equal(answer, answer_one_hot)


def test_a_file_that_is_not_mnist_is_refused(tmp_path):
    """The MNIST file was downloaded, so its checksum is checked before anything in it is used."""
    path = tmp_path / "mnist.npz"
    np.savez(path, x_train=np.zeros((1, 28, 28), dtype=np.uint8), y_train=np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError):
        MNISTImages("train", str(path))


# -------------------------------------------------------------------------------------------------------
# The probes in verify.py
# -------------------------------------------------------------------------------------------------------

def test_a_silenced_head_reads_nothing_until_it_is_restored():
    """verify.py, ablations: a silenced head reads nothing and shines no light; removing the override restores it."""
    diary = diary_with_pages(n_pages=4, batch_size=2)
    seen = torch.randn(2, D_INPUT)
    read_before, _ = diary.read_by_content(seen)

    silence(diary, "read_by_content")
    read, light = diary.read_by_content(seen)
    assert torch.equal(read, torch.zeros(2, VALUE_SIZE))
    assert torch.equal(light, torch.zeros(2, 4))

    delattr(diary, "read_by_content")
    read_after, _ = diary.read_by_content(seen)
    assert torch.equal(read_after, read_before)


def test_lost_accuracy_is_counted_per_kind_of_question():
    """verify.py, ablations: the accuracy a head carries, for pure recall and with operations, kept apart."""
    with_every_head = {}
    without_next = {}
    for n_digits, n_ops in itertools.product(ABLATION_DIGITS, range(4)):
        with_every_head[f"{n_digits},{n_ops}"] = 0.9
        without_next[f"{n_digits},{n_ops}"] = 0.4 if n_ops == 0 else 0.9
    per_seed = [{"none": with_every_head, "read_next": without_next}]

    assert lost_accuracy(per_seed, "read_next", (0,)).tolist() == pytest.approx([0.5])
    assert lost_accuracy(per_seed, "read_next", (1, 2, 3)).tolist() == pytest.approx([0.0])

def test_a_wrong_answer_is_shared_among_the_pages_that_hold_it():
    """verify.py, prediction 3: a wrong answer lands, in equal shares, on every page that holds that digit."""
    digits = np.array([[3, 7, 3, 5]])                                  # four pages
    asked_pages = np.array([1])                                         # the question asks for page 1, which holds 7

    answered_3 = np.zeros((1, 10))
    answered_3[0, 3] = 1.0                                              # pages 0 and 2 hold 3: gaps -1 and +1
    by_gap, nowhere = mass_by_gap(digits, asked_pages, answered_3)
    assert by_gap.tolist() == [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]      # gaps -3, ..., +3
    assert nowhere == 0.0

    answered_9 = np.zeros((1, 10))
    answered_9[0, 9] = 1.0                                              # no page holds 9
    by_gap, nowhere = mass_by_gap(digits, asked_pages, answered_9)
    assert by_gap.sum() == 0.0
    assert nowhere == 1.0
    assert neighbour_share(by_gap) is None                              # no share of nothing, and no NaN in the JSON


def test_slips_never_land_on_the_asked_page():
    """verify.py, Theorem 16: given a slip, the chances over pages add up to one and are 0 on the asked page."""
    clock = Clock()
    trust = clock.trust.double().numpy()
    speeds = clock.speeds.double().numpy()
    asked_pages = np.array([0, 7, 15])
    chances = slip_chances(trust, speeds, asked_pages, n_pages=16)

    assert np.allclose(chances.sum(axis=1), 1.0)
    assert np.all(chances[np.arange(3), asked_pages] == 0.0)

    # The agreement is symmetric, so from page 7 a slip is as likely to land 3 pages back as 3 pages on.
    assert np.isclose(chances[1, 7 - 3], chances[1, 7 + 3])
