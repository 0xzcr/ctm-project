"""
Check the thesis's predictions (its section "What to test next") against the trained models in results/, and run the
extra checks below.

    python verify.py

It needs both variants trained for seeds 0-19 (see the README). The predictions, and what would count against them:

    1. The two-digit drop shrinks   two-digit accuracy well above Box 1's 0.47-0.67      against: staying in that range
    2. Values are distinct          by-number reads decode to the asked-for digit in at least 98% of probes
    3. Slips look designed          wrong answers, by gap, look like the designed clock's slips   against: neighbours
    4. Length                       recall with 5-16 digits stays close to recall with 1-4   against: a fall like LEM's
    5. Every seed reported

Predictions 1, 4 and 5 use the accuracies that train.py saved. Predictions 2 and 3 probe the trained diaries with
pure-recall questions ("index i", no operations), with up to 48 digits (49 pages, inside the clock's horizon of 50).

The extra checks run for the variants that have been trained (see the README). Each one is paired by seed:

    size          the diary against a plain CTM with 107 neurons, which has as many parameters    (ctm_d107)
    latest page   the diary with a fifth head that reads the latest page, against the diary       (ctm_with_diary_latest)
    images        the diary against the plain CTM, both shown real MNIST images      (ctm_with_diary_mnist, ctm_mnist)
                  from scratch, and with the image backbone first trained to tell digits apart and then frozen,
                  trained 4x longer on 5 seeds                                     (..._mnist_pretrained_steps16000)
    ablations     every head of the trained diaries silenced in turn, on train.py's evaluation episodes

The report is printed and saved in results/verification.json.
"""
import argparse
import contextlib
import io
import itertools
import json
import multiprocessing
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from cem.clock import Clock
from cem.memory import FOCUS
from cem.model import build_model
from cem.task import sample_episodes
from train import LONGER_DIGITS, TICKS_PER_ITEM, TRAIN_DIGITS, TRAIN_OPS, accuracy, answer_ticks

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
EARLIER_RESULTS = os.path.expanduser("~/ctm/results/primary")   # the runs behind Box 1, where they exist

BOX1_BEST_TWO_DIGITS = 0.668          # the best two-digit accuracy in Box 1 (the CTM with a long history)
CLOSE_ENOUGH = 0.9                    # prediction 4 holds if 5-16 digits keep at least 90% of the 1-4 accuracy
READ_DECODE_TARGET = 0.98             # prediction 2

ALL_DIGITS = TRAIN_DIGITS + LONGER_DIGITS
PROBE_DIGITS = (2, 4, 8, 16, 32, 48)  # 48 digits and the question make 49 pages, inside the horizon of 50
PROBE_EPISODES = 2000
FIT_EPISODES = 1000                   # episodes for fitting the class codes, kept apart from the probes
FIT_SEED = 30_000
PROBE_SEED = 40_000
N_CLASSES = 10

MARGIN = Clock().margin               # m_H of the design point, for Theorem 5's guarantee

SIZE_MATCHED = "ctm_d107"             # the plain CTM with as many parameters as the CTM with the diary
LATEST = "ctm_with_diary_latest"
EXTRA_VARIANTS = (SIZE_MATCHED, LATEST, "ctm_mnist", "ctm_with_diary_mnist", "ctm_mnist_pretrained_steps16000",
                  "ctm_with_diary_mnist_pretrained_steps16000")

# (name, variant, baseline, the question the pair answers)
COMPARISONS = (
    ("size", "ctm_with_diary", SIZE_MATCHED, "is the diary's gain only its extra parameters?"),
    ("latest page", LATEST, "ctm_with_diary", "were the errors left over from having nowhere to keep the running result?"),
    ("images, from scratch", "ctm_with_diary_mnist", "ctm_mnist",
     "does the diary still help when the digits are real MNIST images?"),
    ("images, pretrained backbone", "ctm_with_diary_mnist_pretrained_steps16000", "ctm_mnist_pretrained_steps16000",
     "does it help once the frozen image backbone can tell the digits apart, with 4x the training?"),
)

ABLATION_DIGITS = (2, 4, 8, 16)
HEADS = {                             # the diary's method for each head, and its name in the report
    "read_by_number": "by number",
    "read_next": "next page",
    "read_previous": "previous page",
    "read_by_content": "content",
    "read_latest": "latest page",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(PROJECT_FOLDER, "results"), help="where train.py saved")
    parser.add_argument("--seeds", type=int, default=20, help="check seeds 0 to this number minus one")
    parser.add_argument("--workers", type=int, default=8, help="seeds checked at once, one CPU thread each")
    return parser.parse_args()


# -------------------------------------------------------------------------------------------------------
# Accuracies saved by train.py
# -------------------------------------------------------------------------------------------------------

def load_reports(folder: str, variant: str, seeds) -> dict:
    """{seed: the accuracy report train.py saved} for one variant, "ctm" or "ctm_with_diary"."""
    reports = {}
    for seed in seeds:
        with open(os.path.join(folder, f"{variant}_seed{seed}.json")) as file:
            reports[seed] = json.load(file)["accuracy"]
    return reports


def load_trained(folder: str, variant: str, seeds) -> dict:
    """load_reports() for the seeds of the variant that have been trained, maybe none: the extra checks are optional."""
    trained = []
    for seed in seeds:
        if os.path.exists(os.path.join(folder, f"{variant}_seed{seed}.json")):
            trained.append(seed)
    return load_reports(folder, variant, trained)


def n_parameters(folder: str, variant: str, seed: int) -> int:
    """The number of parameters of a variant (every seed has the same), as train.py saved it for one seed."""
    with open(os.path.join(folder, f"{variant}_seed{seed}.json")) as file:
        return json.load(file)["parameters"]


def backbone_test_accuracy(folder: str, variant: str, seeds):
    """The mean over seeds of the pretrained backbone's accuracy on MNIST test digits, or None without pretraining."""
    values = []
    for seed in seeds:
        with open(os.path.join(folder, f"{variant}_seed{seed}.json")) as file:
            value = json.load(file).get("pretrained_backbone_test_accuracy")
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def accuracy_by_seed(reports: dict, n_digits: int) -> np.ndarray:
    """(seeds,) accuracy with n_digits digits, averaged over 0-3 operations (the number train.py prints)."""
    values = []
    for report in reports.values():
        values.append(report[str(n_digits)]["mean"])
    return np.array(values)


def mean_over_lengths(reports: dict, lengths) -> np.ndarray:
    """(seeds,) the average of accuracy_by_seed over several numbers of digits."""
    columns = []
    for n_digits in lengths:
        columns.append(accuracy_by_seed(reports, n_digits))
    return np.mean(columns, axis=0)


def mean_with_interval(values: np.ndarray) -> dict:
    """The mean over seeds and its 95% confidence interval (Student t)."""
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "low": mean, "high": mean}
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": mean, "low": float(mean - half_width), "high": float(mean + half_width)}


# -------------------------------------------------------------------------------------------------------
# Is the plain CTM here the CTM of Box 1?
# -------------------------------------------------------------------------------------------------------

def same_as_box1(plain: dict) -> dict:
    """For how many seeds every accuracy cell equals the earlier run behind Box 1 (if those runs are here)."""
    if not os.path.isdir(EARLIER_RESULTS):
        return {"checked": 0, "identical": 0}

    identical = 0
    for seed, report in plain.items():
        with open(os.path.join(EARLIER_RESULTS, f"ctm_s{seed}.json")) as file:
            earlier_grid = json.load(file)["grid"]
        if all_cells_equal(report, earlier_grid):
            identical += 1
    return {"checked": len(plain), "identical": identical}


def all_cells_equal(report: dict, earlier_grid: dict) -> bool:
    """True if every (digits, operations) accuracy in a train.py report equals the earlier run's."""
    for n_digits, row in report.items():
        for n_ops, accuracy in row["by_ops"].items():
            if accuracy != earlier_grid[f"{n_digits},{n_ops}"]["acc"]:
                return False
    return True


# -------------------------------------------------------------------------------------------------------
# Predictions 1 and 4
# -------------------------------------------------------------------------------------------------------

def two_digit_drop(plain: dict, diary: dict) -> dict:
    """Prediction 1: with the diary, two-digit accuracy moves well above Box 1's 0.47-0.67."""
    plain_two = accuracy_by_seed(plain, 2)
    diary_two = accuracy_by_seed(diary, 2)
    diary_summary = mean_with_interval(diary_two)
    return {
        "ctm": mean_with_interval(plain_two),
        "ctm_with_diary": diary_summary,
        "difference": mean_with_interval(diary_two - plain_two),
        "holds": diary_summary["low"] > BOX1_BEST_TWO_DIGITS,
        "rule": f"holds if the diary's 95% interval lies above {BOX1_BEST_TWO_DIGITS}, the best of Box 1",
    }


def length(plain: dict, diary: dict) -> dict:
    """Prediction 4: accuracy with 5-16 digits stays close to accuracy with 1-4 digits."""
    result = {}
    for name, reports in (("ctm", plain), ("ctm_with_diary", diary)):
        short = mean_over_lengths(reports, TRAIN_DIGITS).mean()
        long = mean_over_lengths(reports, LONGER_DIGITS).mean()
        result[name] = {"1-4 digits": float(short), "5-16 digits": float(long), "kept": float(long / short)}

    lem_kept = lem_length_kept()
    diary_kept = result["ctm_with_diary"]["kept"]
    if diary_kept >= CLOSE_ENOUGH:
        verdict = "holds"
    elif lem_kept is not None and diary_kept <= lem_kept:
        verdict = "fails: a fall like LEM's"
    else:
        verdict = "partly: better than LEM, but not close"

    result["by_operations"] = {
        "ctm": accuracy_by_operations(plain),
        "ctm_with_diary": accuracy_by_operations(diary),
    }
    result["lem_kept"] = lem_kept
    result["verdict"] = verdict
    result["rule"] = (f"holds if 5-16 digits keep at least {CLOSE_ENOUGH:.0%} of the 1-4 accuracy; "
                      "fails if they keep no more than LEM did")
    return result


def accuracy_by_operations(reports: dict) -> dict:
    """{n_ops: {n_digits: accuracy, averaged over seeds}}. Shows which questions the errors come from."""
    table = {}
    for n_ops in range(4):
        table[n_ops] = row_of_accuracies(reports, n_ops)
    return table


def row_of_accuracies(reports: dict, n_ops: int) -> dict:
    """{n_digits: accuracy with n_ops operations, averaged over seeds}."""
    row = {}
    for n_digits in TRAIN_DIGITS + LONGER_DIGITS:
        values = [report[str(n_digits)]["by_ops"][str(n_ops)] for report in reports.values()]
        row[n_digits] = float(np.mean(values))
    return row


def lem_length_kept():
    """The share of its 1-4 digit accuracy that LEM (log focus) kept at 5-16 digits, from the earlier runs."""
    if not os.path.isdir(EARLIER_RESULTS):
        return None

    shorts = []
    longs = []
    for seed in range(20):
        with open(os.path.join(EARLIER_RESULTS, f"mem_log_s{seed}.json")) as file:
            curve = json.load(file)["recall_curve"]
        shorts.append(np.mean([curve[str(n)] for n in TRAIN_DIGITS]))
        longs.append(np.mean([curve[str(n)] for n in LONGER_DIGITS]))
    return float(np.mean(longs) / np.mean(shorts))


# -------------------------------------------------------------------------------------------------------
# Probing a trained diary (predictions 2 and 3)
# -------------------------------------------------------------------------------------------------------

def load_diary_model(folder: str, seed: int, variant: str = "ctm_with_diary"):
    """The trained CTM with the diary for one seed, ready to probe. `variant`: ctm_with_diary or LATEST."""
    with contextlib.redirect_stdout(io.StringIO()):          # the upstream code prints its settings
        model = build_model(use_diary=True, seed=seed, latest_head=variant == LATEST)
    weights = torch.load(os.path.join(folder, f"{variant}_seed{seed}.pt"), map_location="cpu")
    model.load_state_dict(weights)
    model.eval()
    return model


def run_and_record(model, x: torch.Tensor, z: torch.Tensor):
    """
    One forward pass that also records what the diary writes and what its by-number head reads. Nothing in the
    model changes. Returns the predictions and certainties, every page as it was written (B, n_writes, d_v),
    and the by-number head's (read, light) on every tick on which it holds a page.
    """
    diary = model.diary
    written = []
    by_number = []
    write_page = diary.write
    read_by_number = diary.read_by_number

    def recording_write(event_synchronisation, seen):
        write_page(event_synchronisation, seen)
        written.append(diary.values[:, -1].clone())                  # the new page, before any edit

    def recording_read_by_number(page_stamps):
        read, light = read_by_number(page_stamps)
        if diary.number_page is not None:
            by_number.append((read.clone(), light.clone()))
        return read, light

    diary.write = recording_write
    diary.read_by_number = recording_read_by_number
    try:
        with torch.no_grad():
            predictions, certainties, _ = model(x, z)
    finally:
        del diary.write                                               # back to the methods of the class
        del diary.read_by_number

    return predictions, certainties, torch.stack(written, dim=1), by_number


def shown_digits(x: torch.Tensor) -> torch.Tensor:
    """The digit on every page: (B, n_digits). Every digit is shown for R ticks, so keep one tick of each."""
    return x[:, ::TICKS_PER_ITEM].argmax(dim=2).flatten(1)


def model_answers(predictions: torch.Tensor, certainties: torch.Tensor) -> torch.Tensor:
    """The model's answers, read out at the most certain answer tick (the read-out of train.accuracy())."""
    predictions, certainties = answer_ticks(predictions, certainties)
    most_certain_tick = certainties[:, 1].argmax(dim=-1)                            # (B,)
    episode = torch.arange(predictions.size(0))
    return predictions[episode, :, most_certain_tick].argmax(dim=-1)               # (B,)


def fit_class_codes(model) -> torch.Tensor:
    """
    One code per digit (assumption A5): the average direction of the pages written for that digit, over
    FIT_EPISODES four-digit episodes that the probes never use. Returns (10, d_v), every code of length 1.
    """
    generator = torch.Generator().manual_seed(FIT_SEED)
    x, z, _ = sample_episodes(FIT_EPISODES, generator, n_digits=4, n_ops=0)
    _, _, written, _ = run_and_record(model, x, z)
    digit_pages = written[:, :4]                                      # (B, 4, d_v)
    digits = shown_digits(x)                                          # (B, 4)

    codes = []
    for digit in range(N_CLASSES):
        pages_with_this_digit = digit_pages[digits == digit]         # (count, d_v)
        codes.append(F.normalize(pages_with_this_digit.mean(dim=0), dim=0))
    return torch.stack(codes)


def decode(vectors: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """The digit whose code points most nearly the same way as each vector.     (..., d_v) -> (...)"""
    cosines = F.normalize(vectors, dim=-1) @ codes.T
    return cosines.argmax(dim=-1)


def largest_code_cosine(codes: torch.Tensor) -> float:
    """rho_v of assumption A5: the largest cosine between the codes of two different digits."""
    cosines = codes @ codes.T
    off_diagonal = cosines[~torch.eye(N_CLASSES, dtype=torch.bool)]
    return off_diagonal.max().item()


# ---- Where wrong answers land (prediction 3) ----------------------------------------------------------

def agreement_by_gap(trust: np.ndarray, speeds: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """kappa(gap) = sum_k w_k cos(omega_k gap), for every gap."""
    return np.cos(np.outer(gaps, speeds)) @ trust


def slip_chances(trust: np.ndarray, speeds: np.ndarray, asked_pages: np.ndarray, n_pages: int) -> np.ndarray:
    """
    Theorem 16: given that the by-number head slips, the chance that it lands on each page is proportional to
    exp(lambda kappa(i - j)), and 0 on the asked page i.     returns (B, n)
    """
    focus = FOCUS * np.log(1 + n_pages)
    pages = np.arange(n_pages)
    chances = np.zeros((len(asked_pages), n_pages))
    for episode, asked in enumerate(asked_pages):
        scores = focus * agreement_by_gap(trust, speeds, asked - pages)
        light = np.exp(scores - scores.max())
        light[asked] = 0.0
        chances[episode] = light / light.sum()
    return chances


def wrong_answer_chances(page_chances: np.ndarray, digits: np.ndarray, right_answers: np.ndarray) -> np.ndarray:
    """
    (B, 10): the chance of each wrong answer if the head had slipped with page_chances, given that the answer
    came out wrong. A slip onto a page that holds the right digit gives the right answer, so it is left out.
    """
    chances = np.zeros((len(digits), N_CLASSES))
    for episode in range(len(digits)):
        chances[episode] = np.bincount(digits[episode], weights=page_chances[episode], minlength=N_CLASSES)
        chances[episode, right_answers[episode]] = 0.0
        total = chances[episode].sum()
        if total > 0:
            chances[episode] = chances[episode] / total
    return chances


def mass_by_gap(digits: np.ndarray, asked_pages: np.ndarray, answer_chances: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Where wrong answers land, by gap from the asked page. A wrong answer w is shared equally among the pages that
    hold the digit w. A wrong answer that no page holds lands nowhere.

    digits:          (B, n) the digit on every page
    asked_pages:     (B,)
    answer_chances:  (B, 10) the chance of every wrong answer (0 for the right one)
    returns:         the mass on gaps -(n - 1), ..., n - 1, and the mass that landed nowhere
    """
    n_episodes, n_pages = digits.shape
    by_gap = np.zeros(2 * n_pages - 1)
    nowhere = 0.0
    for episode in range(n_episodes):
        page_digits = digits[episode]
        pages_holding = np.bincount(page_digits, minlength=N_CLASSES)        # how many pages hold each digit
        chances = answer_chances[episode]
        nowhere += chances[pages_holding == 0].sum()

        share_per_page = chances[page_digits] / pages_holding[page_digits]   # (n,)
        gaps = np.arange(n_pages) - asked_pages[episode]
        by_gap[gaps + n_pages - 1] += share_per_page
    return by_gap, nowhere


def where_wrong_answers_land(model, digits, asked_pages, right_answers, answers) -> dict:
    """
    The model's wrong answers by gap, next to where they would land if they were slips of the designed clock,
    slips of an equal-trust clock (neighbours), or random wrong digits (chance). Only the episodes the model got
    wrong are used, so all four are about the same questions.
    """
    wrong = answers != right_answers
    digits = digits[wrong].numpy()
    asked_pages = asked_pages[wrong].numpy()
    right_answers = right_answers[wrong].numpy()
    answers = answers[wrong].numpy()
    n_wrong, n_pages = digits.shape

    trust = model.diary.clock.trust.double().numpy()
    speeds = model.diary.clock.speeds.double().numpy()
    equal_trust = np.full_like(trust, 1.0 / len(trust))

    observed = np.zeros((n_wrong, N_CLASSES))
    observed[np.arange(n_wrong), answers] = 1.0

    random_wrong_digit = np.full((n_wrong, N_CLASSES), 1.0 / (N_CLASSES - 1))
    random_wrong_digit[np.arange(n_wrong), right_answers] = 0.0

    designed = wrong_answer_chances(slip_chances(trust, speeds, asked_pages, n_pages), digits, right_answers)
    neighbours = wrong_answer_chances(slip_chances(equal_trust, speeds, asked_pages, n_pages), digits, right_answers)

    landed = {}
    for name, chances in (("model", observed), ("designed", designed), ("neighbours", neighbours),
                          ("chance", random_wrong_digit)):
        by_gap, nowhere = mass_by_gap(digits, asked_pages, chances)
        landed[name] = {"by_gap": by_gap, "nowhere": nowhere}
    landed["n_wrong"] = n_wrong
    return landed


def neighbour_share(by_gap: np.ndarray):
    """
    Of the wrong answers that landed on a page, the share that landed on a neighbour of the asked page (gap -1 or
    +1). None if none landed on a page: there is no share of nothing, and JSON has no NaN.
    """
    on_pages = by_gap.sum()
    if on_pages == 0:
        return None
    middle = len(by_gap) // 2
    return float((by_gap[middle - 1] + by_gap[middle + 1]) / on_pages)


def shares(by_gap: np.ndarray):
    """The mass on every gap as shares that add up to one, or None if no wrong answer landed on a page."""
    on_pages = by_gap.sum()
    if on_pages == 0:
        return None
    return (by_gap / on_pages).round(4).tolist()


def distance(first: np.ndarray, second: np.ndarray) -> float:
    """Total variation distance between two distributions over gaps: 0 means the same, 1 means no overlap."""
    return float(0.5 * np.abs(first / first.sum() - second / second.sum()).sum())


# ---- One model, one length ----------------------------------------------------------------------------

def probe(model, codes: torch.Tensor, n_digits: int) -> dict:
    """Pure-recall probes of one trained diary with n_digits digits."""
    generator = torch.Generator().manual_seed(PROBE_SEED + n_digits)
    x, z, right_answers = sample_episodes(PROBE_EPISODES, generator, n_digits, n_ops=0)
    predictions, certainties, written, by_number = run_and_record(model, x, z)

    digits = shown_digits(x)                                          # (B, n)
    asked_pages = z[:, 0]                                             # (B,) the question "index i" names page i
    episode = torch.arange(PROBE_EPISODES)
    answers = model_answers(predictions, certainties)

    # The by-number head's first read comes at the first tick of the question, before anything is crossed off.
    # Its last read comes at the last answer tick, after the question has been crossed off three times.
    first_read, first_light = by_number[0]
    last_read, _ = by_number[-1]

    return {
        "accuracy": (answers == right_answers).float().mean().item(),
        "light_on_asked_page": first_light[episode, asked_pages].min().item(),
        "pages_decoded": (decode(written[:, :n_digits], codes) == digits).float().mean().item(),
        "first_read_decoded": (decode(first_read, codes) == right_answers).float().mean().item(),
        "last_read_decoded": (decode(last_read, codes) == right_answers).float().mean().item(),
        "landed": where_wrong_answers_land(model, digits, asked_pages, right_answers, answers),
    }


def gather_probes(checked: list) -> dict:
    """The probes of every seed (from check_every_seed), by length, every seed kept."""
    by_length = {}
    for n_digits in PROBE_DIGITS:
        by_length[n_digits] = [result["probes"][n_digits] for result in checked]
    code_cosines = [result["largest_code_cosine"] for result in checked]
    return {"by_length": by_length, "largest_code_cosine": code_cosines}


def summarise_probes(probes: dict) -> dict:
    """Predictions 2 and 3, and Theorem 5 inside the trained models, from the probes of every seed."""
    summary = {"largest_code_cosine": float(np.max(probes["largest_code_cosine"]))}
    for n_digits, per_seed in probes["by_length"].items():
        summary[n_digits] = summarise_length(n_digits, per_seed)

    lowest_decoding = min(summary[n_digits]["first_read_decoded"]["mean"] for n_digits in PROBE_DIGITS)
    summary["prediction_2"] = {
        "holds": lowest_decoding >= READ_DECODE_TARGET,
        "rule": f"holds if, at every length, by-number reads decode to the asked digit in at least "
                f"{READ_DECODE_TARGET:.0%} of probes (mean over seeds)",
    }

    # Only long diaries tell the clocks apart: with 2 digits every slip is onto a neighbour.
    closest = summary[max(PROBE_DIGITS)]["closest_to_model"]
    if closest == "designed":
        verdict = "holds"
    elif closest == "neighbours":
        verdict = "against: the wrong answers land on neighbours"
    else:
        verdict = "does not hold: the wrong answers look like random wrong digits, not slips of any clock"
    summary["prediction_3"] = {
        "verdict": verdict,
        "rule": f"with {max(PROBE_DIGITS)} digits, the model's wrong answers must land most like the designed "
                "clock's slips (distance over gaps, plus answers that no page holds)",
    }
    return summary


def summarise_length(n_digits: int, per_seed: list) -> dict:
    """Everything the probes found with n_digits digits, over all seeds."""
    guarantee = 1 / (1 + (n_digits - 1) * (1 + n_digits) ** (-FOCUS * MARGIN))
    first_read_decoded = np.array([result["first_read_decoded"] for result in per_seed])
    n_wrong_answers = int(sum(result["landed"]["n_wrong"] for result in per_seed))

    slips = {}
    closest = None
    if n_wrong_answers > 0:
        landed_by_name = {}
        for name in ("model", "designed", "neighbours", "chance"):
            landed_by_name[name] = pool_over_seeds(per_seed, name)

        landed_by_model = with_nowhere(landed_by_name["model"])
        for name, pooled in landed_by_name.items():
            slips[name] = {
                "neighbour_share": neighbour_share(pooled["by_gap"]),
                "share_landed_nowhere": pooled["nowhere"] / (pooled["nowhere"] + pooled["by_gap"].sum()),
                "distance_from_model": distance(landed_by_model, with_nowhere(pooled)),
                "by_gap": shares(pooled["by_gap"]),
            }
        closest = min(("designed", "neighbours", "chance"), key=lambda name: slips[name]["distance_from_model"])

    return {
        "recall_accuracy": float(np.mean([result["accuracy"] for result in per_seed])),
        "least_light_on_asked_page": float(np.min([result["light_on_asked_page"] for result in per_seed])),
        "theorem_5_guarantee": guarantee,
        "pages_decoded": float(np.mean([result["pages_decoded"] for result in per_seed])),
        "first_read_decoded": mean_with_interval(first_read_decoded),
        "first_read_decoded_by_seed": first_read_decoded.round(4).tolist(),
        "last_read_decoded": float(np.mean([result["last_read_decoded"] for result in per_seed])),
        "n_wrong_answers": n_wrong_answers,
        "slips": slips,
        "closest_to_model": closest,
    }


def with_nowhere(landed: dict) -> np.ndarray:
    """The mass by gap, plus one more bin for the wrong answers that no page holds (no clock slips there)."""
    return np.append(landed["by_gap"], landed["nowhere"])


def pool_over_seeds(per_seed: list, name: str) -> dict:
    """Add up where the wrong answers of every seed landed, for one of model, designed, neighbours, chance."""
    by_gap = 0.0
    nowhere = 0.0
    for result in per_seed:
        by_gap = by_gap + result["landed"][name]["by_gap"]
        nowhere = nowhere + result["landed"][name]["nowhere"]
    return {"by_gap": by_gap, "nowhere": nowhere}


# -------------------------------------------------------------------------------------------------------
# Silencing one head at a time
# -------------------------------------------------------------------------------------------------------

def silence(diary, head: str):
    """Silence one head: it reads nothing and shines no light, so it adds nothing to the read and edits nothing."""
    setattr(diary, head, lambda *arguments: diary.nothing())


def accuracy_grid(model) -> dict:
    """{"n_digits,n_ops": accuracy} on train.py's evaluation episodes, for ABLATION_DIGITS and 0-3 operations."""
    grid = {}
    for n_digits, n_ops in itertools.product(ABLATION_DIGITS, TRAIN_OPS):
        grid[f"{n_digits},{n_ops}"] = accuracy(model, n_digits, n_ops, "cpu")
    return grid


def ablate(model) -> dict:
    """
    Accuracy with each head silenced in turn, and with none ("none"): {head: {"n_digits,n_ops": accuracy}}.
    The episodes are the ones train.py evaluated on, so the accuracies with no head silenced must equal its own.
    """
    heads = ["read_by_number", "read_next", "read_previous", "read_by_content"]
    if model.diary.latest_head:
        heads.append("read_latest")

    grids = {"none": accuracy_grid(model)}
    for head in heads:
        silence(model.diary, head)
        try:
            grids[head] = accuracy_grid(model)
        finally:
            delattr(model.diary, head)                                # back to the method of the class
    return grids


def lost_accuracy(per_seed: list, head: str, op_counts) -> np.ndarray:
    """(seeds,) accuracy with every head minus accuracy with `head` silenced, averaged over the cells asked for."""
    cells = [f"{n_digits},{n_ops}" for n_digits, n_ops in itertools.product(ABLATION_DIGITS, op_counts)]
    lost = []
    for grids in per_seed:
        lost.append(np.mean([grids["none"][cell] - grids[head][cell] for cell in cells]))
    return np.array(lost)


def unsilenced_matches_saved(grids: dict, report: dict) -> bool:
    """True if the accuracies with no head silenced equal, cell for cell, the ones train.py saved."""
    for cell, value in grids["none"].items():
        n_digits, n_ops = cell.split(",")
        if value != report[n_digits]["by_ops"][n_ops]:
            return False
    return True


def summarise_heads(per_seed: list, saved: dict) -> dict:
    """What silencing each head costs, for pure recall (0 operations) and with 1-3 operations, over all seeds."""
    matching = 0
    for grids, report in zip(per_seed, saved.values()):
        matching += unsilenced_matches_saved(grids, report)

    summary = {"seeds_matching_train_py": matching, "heads": {}}
    for head in per_seed[0]:
        if head == "none":
            continue
        summary["heads"][head] = {
            "pure recall": mean_with_interval(lost_accuracy(per_seed, head, (0,))),
            "with operations": mean_with_interval(lost_accuracy(per_seed, head, (1, 2, 3))),
        }
    return summary


def summarise_ablations(checked: list, variants: dict) -> dict:
    """summarise_heads() for every diary variant that was ablated."""
    summary = {}
    for variant in checked[0]["ablations"]:
        per_seed = [result["ablations"][variant] for result in checked]
        summary[variant] = summarise_heads(per_seed, variants[variant])
    return summary


# -------------------------------------------------------------------------------------------------------
# Every seed's trained diaries, probed and ablated, several seeds at once
# -------------------------------------------------------------------------------------------------------

def check_one_seed(job: tuple) -> dict:
    """
    The probes of one seed's trained diary, and the ablations of its diary variants. Runs in a worker process
    with one CPU thread, as training did, so that the unsilenced accuracies come out exactly as train.py saved them.
    """
    folder, seed, ablated_variants = job
    torch.set_num_threads(1)

    model = load_diary_model(folder, seed)
    codes = fit_class_codes(model)
    probes = {}
    for n_digits in PROBE_DIGITS:
        probes[n_digits] = probe(model, codes, n_digits)

    ablations = {}
    for variant in ablated_variants:
        ablations[variant] = ablate(load_diary_model(folder, seed, variant))

    print(f"    checked seed {seed}", flush=True)
    return {"largest_code_cosine": largest_code_cosine(codes), "probes": probes, "ablations": ablations}


def check_every_seed(folder: str, seeds, workers: int, ablated_variants: list) -> list:
    """check_one_seed() for every seed, `workers` seeds at a time. The results come back in the order of the seeds."""
    jobs = [(folder, seed, ablated_variants) for seed in seeds]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return pool.map(check_one_seed, jobs, chunksize=1)


# -------------------------------------------------------------------------------------------------------
# The extra variants, each against its baseline
# -------------------------------------------------------------------------------------------------------

def accuracy_with_ops(reports: dict, n_ops: int) -> np.ndarray:
    """(seeds,) accuracy with n_ops operations, averaged over every number of digits evaluated."""
    values = []
    for report in reports.values():
        values.append(np.mean([report[str(n_digits)]["by_ops"][str(n_ops)] for n_digits in ALL_DIGITS]))
    return np.array(values)


def kept(reports: dict) -> float:
    """The share of its 1-4 digit accuracy that a variant keeps at 5-16 digits (means over seeds)."""
    return float(mean_over_lengths(reports, LONGER_DIGITS).mean() / mean_over_lengths(reports, TRAIN_DIGITS).mean())


def compare(variant: dict, baseline: dict) -> dict:
    """
    A variant against its baseline, paired by seed: the difference in accuracy (variant minus baseline) with two
    digits, with 1-4 and with 5-16 digits, and with each number of operations; on how many seeds the variant is
    better over all lengths; and how much of its 1-4 digit accuracy each one keeps at 5-16 digits.
    """
    differences = {
        "2 digits": accuracy_by_seed(variant, 2) - accuracy_by_seed(baseline, 2),
        "1-4 digits": mean_over_lengths(variant, TRAIN_DIGITS) - mean_over_lengths(baseline, TRAIN_DIGITS),
        "5-16 digits": mean_over_lengths(variant, LONGER_DIGITS) - mean_over_lengths(baseline, LONGER_DIGITS),
    }
    for n_ops in TRAIN_OPS:
        differences[f"{n_ops} ops"] = accuracy_with_ops(variant, n_ops) - accuracy_with_ops(baseline, n_ops)

    result = {"difference": {}}
    for name, values in differences.items():
        result["difference"][name] = mean_with_interval(values)

    overall = mean_over_lengths(variant, ALL_DIGITS) - mean_over_lengths(baseline, ALL_DIGITS)
    result["seeds_better"] = int((overall > 0).sum())
    result["kept"] = {"variant": kept(variant), "baseline": kept(baseline)}
    return result


def comparisons(folder: str, variants: dict) -> dict:
    """compare() for every pair in COMPARISONS whose two variants have been trained, on the seeds both have."""
    result = {}
    for name, variant, baseline, question in COMPARISONS:
        if variant not in variants or baseline not in variants:
            continue
        seeds = [seed for seed in variants[variant] if seed in variants[baseline]]
        if not seeds:
            continue
        result[name] = {
            "variant": variant,
            "baseline": baseline,
            "question": question,
            "seeds": len(seeds),
            "parameters": {"variant": n_parameters(folder, variant, seeds[0]),
                           "baseline": n_parameters(folder, baseline, seeds[0])},
            "backbone_test_accuracy": {"variant": backbone_test_accuracy(folder, variant, seeds),
                                       "baseline": backbone_test_accuracy(folder, baseline, seeds)},
            **compare(only_seeds(variants[variant], seeds), only_seeds(variants[baseline], seeds)),
        }
    return result


def only_seeds(reports: dict, seeds: list) -> dict:
    """The reports of the given seeds only, in that order, so that two variants pair up seed by seed."""
    return {seed: reports[seed] for seed in seeds}


def accuracy_by_length(variants: dict) -> dict:
    """{variant: mean_by_length()} for every trained variant."""
    table = {}
    for name, reports in variants.items():
        table[name] = mean_by_length(reports)
    return table


def mean_by_length(reports: dict) -> dict:
    """{n_digits: accuracy averaged over 0-3 operations and seeds}, and the number of seeds."""
    row = {"seeds": len(reports)}
    for n_digits in ALL_DIGITS:
        row[n_digits] = float(accuracy_by_seed(reports, n_digits).mean())
    return row


# -------------------------------------------------------------------------------------------------------
# Prediction 5 and the report
# -------------------------------------------------------------------------------------------------------

def every_seed(plain: dict, diary: dict) -> list:
    """Prediction 5: every seed's accuracy, both variants, at a few lengths."""
    rows = []
    for seed in plain:
        row = {"seed": seed}
        for n_digits in (1, 2, 4, 8, 16):
            row[f"ctm {n_digits}"] = plain[seed][str(n_digits)]["mean"]
            row[f"diary {n_digits}"] = diary[seed][str(n_digits)]["mean"]
        rows.append(row)
    return rows


def interval_text(summary: dict) -> str:
    return f"{summary['mean']:.3f} [{summary['low']:.3f}, {summary['high']:.3f}]"


def signed_interval_text(summary: dict) -> str:
    return f"{summary['mean']:+.3f} [{summary['low']:+.3f}, {summary['high']:+.3f}]"


def print_report(report: dict):
    box1 = report["same_as_box1"]
    print(f"\nThe plain CTM here is Box 1's CTM: {box1['identical']} of {box1['checked']} seeds identical in every cell")

    first = report["prediction_1"]
    print("\nPrediction 1, the two-digit drop shrinks (mean over seeds, 95% interval)")
    print(f"    CTM                    {interval_text(first['ctm'])}")
    print(f"    CTM with the diary     {interval_text(first['ctm_with_diary'])}")
    print(f"    difference, paired     {interval_text(first['difference'])}")
    print(f"    {'HOLDS' if first['holds'] else 'DOES NOT HOLD'}: {first['rule']}")

    probes = report["probes"]
    print("\nPrediction 2, values are distinct (pure recall; codes fitted on separate four-digit episodes)")
    print(f"    largest cosine between two digits' codes (rho_v): {probes['largest_code_cosine']:.3f}")
    for n_digits in PROBE_DIGITS:
        found = probes[n_digits]
        print(f"    {n_digits:2d} digits   pages decode {found['pages_decoded']:.3f}   "
              f"first read decodes {interval_text(found['first_read_decoded'])}   "
              f"last read {found['last_read_decoded']:.3f}   model answers {found['recall_accuracy']:.3f}")
    second = probes["prediction_2"]
    print(f"    {'HOLDS' if second['holds'] else 'DOES NOT HOLD'}: {second['rule']}")

    print("\nTheorem 5 inside the trained models: least light on the asked page, at the first question tick")
    for n_digits in PROBE_DIGITS:
        found = probes[n_digits]
        print(f"    {n_digits:2d} digits   {found['least_light_on_asked_page']:.6f}   "
              f"(guarantee {found['theorem_5_guarantee']:.6f})")

    print("\nPrediction 3, slips look designed: where the wrong answers land (pooled over seeds)")
    for n_digits in PROBE_DIGITS[1:]:                         # with 2 digits every slip is onto a neighbour
        found = probes[n_digits]
        print(f"    {n_digits:2d} digits, {found['n_wrong_answers']} wrong answers; closest: {found['closest_to_model']}")
        for name, slips in found["slips"].items():
            on_a_neighbour = "  -  "                          # no wrong answer landed on a page
            if slips["neighbour_share"] is not None:
                on_a_neighbour = f"{slips['neighbour_share']:.3f}"
            print(f"        {name:10s}  on a neighbour {on_a_neighbour}   "
                  f"on no page {slips['share_landed_nowhere']:.3f}   "
                  f"distance from the model {slips['distance_from_model']:.3f}")
    third = probes["prediction_3"]
    print(f"    {third['verdict'].upper()}: {third['rule']}")

    fourth = report["prediction_4"]
    print("\nPrediction 4, length (mean accuracy over 0-3 operations and seeds)")
    for name in ("ctm", "ctm_with_diary"):
        row = fourth[name]
        print(f"    {name:15s} 1-4 digits {row['1-4 digits']:.3f}   5-16 digits {row['5-16 digits']:.3f}   "
              f"kept {row['kept']:.0%}")
    if fourth["lem_kept"] is not None:
        print(f"    LEM (log focus) kept {fourth['lem_kept']:.0%}")
    print(f"    {fourth['verdict'].upper()}: {fourth['rule']}")

    all_lengths = TRAIN_DIGITS + LONGER_DIGITS
    print("    accuracy by number of operations (mean over seeds), with 1 to 16 digits:")
    print("                      " + "".join(f"{n_digits:>7d}" for n_digits in all_lengths))
    for name, table in fourth["by_operations"].items():
        for n_ops, row in table.items():
            print(f"    {name:15s} {n_ops} ops" + "".join(f"{row[n_digits]:7.3f}" for n_digits in all_lengths))

    print("\nPrediction 5, every seed (mean accuracy over 0-3 operations; ctm n / diary n = with n digits)")
    lengths = (1, 2, 4, 8, 16)
    header = "    seed"
    for n_digits in lengths:
        header += f"{'ctm ' + str(n_digits):>10}{'diary ' + str(n_digits):>10}"
    print(header)
    for row in report["prediction_5"]:
        line = f"    {row['seed']:4d}"
        for n_digits in lengths:
            line += f"{row[f'ctm {n_digits}']:>10.3f}{row[f'diary {n_digits}']:>10.3f}"
        print(line)


def print_extra_checks(report: dict):
    print("\nEvery variant: mean accuracy over 0-3 operations and seeds, by number of digits")
    print(" " * 50 + "".join(f"{n_digits:>7d}" for n_digits in ALL_DIGITS))
    for name, row in report["accuracy_by_length"].items():
        print(f"    {name:42s}{row['seeds']:2d} seeds" + "".join(f"{row[n_digits]:7.3f}" for n_digits in ALL_DIGITS))

    for name, pair in report["comparisons"].items():
        parameters = pair["parameters"]
        print(f"\n{name.capitalize()}: {pair['variant']} ({parameters['variant']:,} parameters) against "
              f"{pair['baseline']} ({parameters['baseline']:,}), {pair['seeds']} paired seeds. {pair['question']}")
        print("    difference in accuracy, paired by seed (95% interval):")
        for part, summary in pair["difference"].items():
            print(f"        {part:12s} {signed_interval_text(summary)}")
        print(f"    better over all lengths on {pair['seeds_better']} of {pair['seeds']} seeds; "
              f"keeps {pair['kept']['variant']:.0%} of its 1-4 digit accuracy at 5-16 digits "
              f"({pair['baseline']}: {pair['kept']['baseline']:.0%})")
        backbones = pair["backbone_test_accuracy"]
        if backbones["variant"] is not None:
            print(f"    the pretrained backbones name {backbones['variant']:.3f} and {backbones['baseline']:.3f} of "
                  "MNIST's test digits (mean over seeds)")

    lengths = f"{ABLATION_DIGITS[0]}-{ABLATION_DIGITS[-1]} digits"
    for variant, summary in report["ablations"].items():
        print(f"\nAblations of {variant}: accuracy lost when one head is silenced ({lengths}, mean over seeds, "
              f"95% interval)")
        print(f"    with no head silenced, the accuracies equal train.py's on {summary['seeds_matching_train_py']} "
              f"of {len(report['prediction_5'])} seeds")
        for head, lost in summary["heads"].items():
            print(f"    {HEADS[head]:14s}  pure recall {signed_interval_text(lost['pure recall'])}   "
                  f"with 1-3 operations {signed_interval_text(lost['with operations'])}")


def main():
    args = parse_arguments()
    seeds = range(args.seeds)
    plain = load_reports(args.results, "ctm", seeds)
    diary = load_reports(args.results, "ctm_with_diary", seeds)

    variants = {"ctm": plain, "ctm_with_diary": diary}
    for variant in EXTRA_VARIANTS:
        reports = load_trained(args.results, variant, seeds)
        if reports:
            variants[variant] = reports

    ablated_variants = ["ctm_with_diary"]
    if len(variants.get(LATEST, {})) == len(seeds):
        ablated_variants.append(LATEST)

    print("probing the trained diaries and silencing their heads")
    checked = check_every_seed(args.results, seeds, args.workers, ablated_variants)

    report = {
        "same_as_box1": same_as_box1(plain),
        "prediction_1": two_digit_drop(plain, diary),
        "probes": summarise_probes(gather_probes(checked)),
        "prediction_4": length(plain, diary),
        "prediction_5": every_seed(plain, diary),
        "accuracy_by_length": accuracy_by_length(variants),
        "comparisons": comparisons(args.results, variants),
        "ablations": summarise_ablations(checked, variants),
    }
    print_report(report)
    print_extra_checks(report)

    path = os.path.join(args.results, "verification.json")
    with open(path, "w") as file:
        json.dump(report, file, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
