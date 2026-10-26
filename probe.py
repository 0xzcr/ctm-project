"""
Look inside the trained diaries. Loads every seed's trained models, eight seeds at a time; about ten minutes.

    python probe.py

It probes the trained diaries with pure-recall questions ("index i", no operations) of up to 48 digits (49 pages,
inside the clock's horizon of 50), and checks two of the thesis's predictions (its section "What to test next"):

    2. Values are distinct    by-number reads decode to the asked-for digit in at least 98% of probes
    3. Slips look designed    wrong answers, by gap, look like the designed clock's slips    against: neighbours

It also checks Theorem 5 inside the trained models, and silences every head of the trained diaries in turn, on
train.py's evaluation episodes (ablations). The report is printed and saved in results/probes.json.
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

from cem.diary import FOCUS, HEADS, Clock
from cem.model import build_model
from cem.task import sample_episodes
from compare import accuracies, load_runs, mean_with_interval
from train import RESULTS_FOLDER, TICKS_PER_ITEM, TRAIN_OPS, accuracy, answer_ticks

READ_DECODE_TARGET = 0.98             # prediction 2
PROBE_DIGITS = (2, 4, 8, 16, 32, 48)  # 48 digits and the question make 49 pages, inside the horizon of 50
PROBE_EPISODES = 2000
FIT_EPISODES = 1000                   # episodes for fitting the class codes, kept apart from the probes
FIT_SEED = 30_000
PROBE_SEED = 40_000
N_CLASSES = 10
MARGIN = Clock().margin               # m_H of the design point, for Theorem 5's guarantee

LATEST = "ctm_with_diary_latest"
ABLATION_DIGITS = (2, 4, 8, 16)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=RESULTS_FOLDER, help="where train.py saved")
    parser.add_argument("--seeds", type=int, default=20, help="check seeds 0 to this number minus one")
    parser.add_argument("--workers", type=int, default=8, help="seeds checked at once, one CPU thread each")
    return parser.parse_args()


# -------------------------------------------------------------------------------------------------------
# Running a trained diary
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
    One forward pass that also records every page as it was written (B, n_writes, d_v), and the by-number head's
    (read, light) on every tick on which it holds a page.
    """
    model.diary.recording = {"pages": [], "by_number": []}
    with torch.no_grad():
        predictions, certainties, _ = model(x, z)
    recording = model.diary.recording
    model.diary.recording = None
    return predictions, certainties, torch.stack(recording["pages"], dim=1), recording["by_number"]


def shown_digits(x: torch.Tensor) -> torch.Tensor:
    """The digit on every page: (B, n_digits). Every digit is shown for R ticks, so keep one tick of each."""
    return x[:, ::TICKS_PER_ITEM].argmax(dim=2).flatten(1)


def model_answers(predictions: torch.Tensor, certainties: torch.Tensor) -> torch.Tensor:
    """The model's answers, read out at the most certain answer tick (the read-out of train.accuracy())."""
    predictions, certainties = answer_ticks(predictions, certainties)
    most_certain_tick = certainties[:, 1].argmax(dim=-1)                            # (B,)
    episode = torch.arange(predictions.size(0))
    return predictions[episode, :, most_certain_tick].argmax(dim=-1)               # (B,)


# -------------------------------------------------------------------------------------------------------
# Prediction 2: what the pages and the reads hold
# -------------------------------------------------------------------------------------------------------

def fit_class_codes(model) -> torch.Tensor:
    """
    One code per digit (assumption A5): the average direction of the pages written for that digit, over FIT_EPISODES
    four-digit episodes that the probes never use. Returns (10, d_v), every code of length 1.
    """
    x, z, _ = sample_episodes(FIT_EPISODES, torch.Generator().manual_seed(FIT_SEED), n_digits=4, n_ops=0)
    _, _, written, _ = run_and_record(model, x, z)
    digit_pages = written[:, :4]
    digits = shown_digits(x)

    codes = []
    for digit in range(N_CLASSES):
        pages_with_this_digit = digit_pages[digits == digit]                      # (count, d_v)
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


# -------------------------------------------------------------------------------------------------------
# Prediction 3: where the wrong answers land
# -------------------------------------------------------------------------------------------------------

def slip_chances(trust: np.ndarray, speeds: np.ndarray, asked_pages: np.ndarray, n_pages: int) -> np.ndarray:
    """
    Theorem 16: given that the by-number head slips, the chance that it lands on each page is proportional to
    exp(lambda kappa(i - j)), and 0 on the asked page i.     returns (B, n)
    """
    focus = FOCUS * np.log(1 + n_pages)
    pages = np.arange(n_pages)
    chances = np.zeros((len(asked_pages), n_pages))
    for episode, asked in enumerate(asked_pages):
        agreement = np.cos(np.outer(asked - pages, speeds)) @ trust              # kappa(i - j) for every page j
        scores = focus * agreement
        light = np.exp(scores - scores.max())
        light[asked] = 0.0
        chances[episode] = light / light.sum()
    return chances


def wrong_answer_chances(page_chances: np.ndarray, digits: np.ndarray, right_answers: np.ndarray) -> np.ndarray:
    """
    (B, 10): the chance of each wrong answer if the head had slipped with page_chances, given that the answer came
    out wrong. A slip onto a page that holds the right digit gives the right answer, so it is left out.
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
    Where wrong answers land, by gap from the asked page. A wrong answer w is shared equally among the pages that hold
    the digit w; a wrong answer that no page holds lands nowhere. digits (B, n), asked_pages (B,), answer_chances
    (B, 10) -> the mass on gaps -(n - 1), ..., n - 1, and the mass that landed nowhere.
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
    The model's wrong answers by gap, next to where they would land if they were slips of the designed clock, slips of
    an equal-trust clock (neighbours), or random wrong digits (chance). Only the episodes the model got wrong are used,
    so all four are about the same questions.
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
    Of the wrong answers that landed on a page, the share that landed on a neighbour of the asked page (gap -1 or +1).
    None if none landed on a page: there is no share of nothing, and JSON has no NaN.
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


# -------------------------------------------------------------------------------------------------------
# One trained diary, one length
# -------------------------------------------------------------------------------------------------------

def probe(model, codes: torch.Tensor, n_digits: int) -> dict:
    """Pure-recall probes of one trained diary with n_digits digits."""
    generator = torch.Generator().manual_seed(PROBE_SEED + n_digits)
    x, z, right_answers = sample_episodes(PROBE_EPISODES, generator, n_digits, n_ops=0)
    predictions, certainties, written, by_number = run_and_record(model, x, z)

    digits = shown_digits(x)
    asked_pages = z[:, 0]                                             # the question "index i" names page i
    episode = torch.arange(PROBE_EPISODES)
    answers = model_answers(predictions, certainties)

    # The by-number head's first read comes at the first tick of the question, before anything is crossed off. Its
    # last read comes at the last answer tick, after the question has been crossed off three times.
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


# -------------------------------------------------------------------------------------------------------
# Silencing one head at a time (ablations)
# -------------------------------------------------------------------------------------------------------

def accuracy_grid(model) -> dict:
    """{"n_digits,n_ops": accuracy} on train.py's evaluation episodes, for ABLATION_DIGITS and 0-3 operations."""
    grid = {}
    for n_digits, n_ops in itertools.product(ABLATION_DIGITS, TRAIN_OPS):
        grid[f"{n_digits},{n_ops}"] = accuracy(model, n_digits, n_ops, "cpu")
    return grid


def ablate(model) -> dict:
    """
    Accuracy with each head silenced in turn, and with none ("none"): {head: {"n_digits,n_ops": accuracy}}. The
    episodes are the ones train.py evaluated on, so the accuracies with no head silenced must equal its own.
    """
    heads = list(HEADS[:4])
    if model.diary.latest_head:
        heads.append("latest page")

    grids = {"none": accuracy_grid(model)}
    for head in heads:
        model.diary.silenced = {head}
        grids[head] = accuracy_grid(model)
    model.diary.silenced = set()
    return grids


def lost_accuracy(per_seed: list, head: str, op_counts) -> np.ndarray:
    """(seeds,) mean_lost() of every seed."""
    lost = []
    for grids in per_seed:
        lost.append(mean_lost(grids, head, op_counts))
    return np.array(lost)


def mean_lost(grids: dict, head: str, op_counts) -> float:
    """One seed's accuracy with every head minus its accuracy with `head` silenced, averaged over the cells asked."""
    lost = []
    for n_digits, n_ops in itertools.product(ABLATION_DIGITS, op_counts):
        cell = f"{n_digits},{n_ops}"
        lost.append(grids["none"][cell] - grids[head][cell])
    return np.mean(lost)


def matches_saved(grid: dict, report: dict) -> bool:
    """True if every accuracy in an accuracy_grid() equals, cell for cell, the one train.py saved."""
    for cell, value in grid.items():
        n_digits, n_ops = cell.split(",")
        if value != report[n_digits]["by_ops"][n_ops]:
            return False
    return True


def summarise_heads(per_seed: list, saved: dict) -> dict:
    """What silencing each head costs, for pure recall (0 operations) and with 1-3 operations, over all seeds."""
    matching = 0
    for grids, report in zip(per_seed, saved.values()):
        matching += matches_saved(grids["none"], report)

    summary = {"seeds_matching_train_py": matching, "heads": {}}
    for head in per_seed[0]:
        if head == "none":
            continue
        summary["heads"][head] = {
            "pure recall": mean_with_interval(lost_accuracy(per_seed, head, (0,))),
            "with operations": mean_with_interval(lost_accuracy(per_seed, head, (1, 2, 3))),
        }
    return summary


# -------------------------------------------------------------------------------------------------------
# Every seed, several at once
# -------------------------------------------------------------------------------------------------------

def check_one_seed(job: tuple) -> dict:
    """
    The probes of one seed's trained diary, and the ablations of its diary variants. Runs in a worker process with one
    CPU thread, as training did, so that the accuracies with no head silenced come out exactly as train.py saved them.
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


def of_every_seed(checked: list, part: str, key) -> list:
    """What check_one_seed() found for one part ("probes" or "ablations") and one key, for every seed."""
    found = []
    for result in checked:
        found.append(result[part][key])
    return found


def pool_over_seeds(per_seed: list, name: str) -> dict:
    """Add up where the wrong answers of every seed landed, for one of model, designed, neighbours, chance."""
    by_gap = 0.0
    nowhere = 0.0
    for result in per_seed:
        by_gap = by_gap + result["landed"][name]["by_gap"]
        nowhere = nowhere + result["landed"][name]["nowhere"]
    return {"by_gap": by_gap, "nowhere": nowhere}


def summarise_length(n_digits: int, per_seed: list) -> dict:
    """Everything the probes found with n_digits digits, over all seeds."""
    guarantee = 1 / (1 + (n_digits - 1) * (1 + n_digits) ** (-FOCUS * MARGIN))
    first_read_decoded = np.array([result["first_read_decoded"] for result in per_seed])
    n_wrong_answers = int(sum(result["landed"]["n_wrong"] for result in per_seed))

    slips = {}
    closest = None
    if n_wrong_answers > 0:
        # Each distribution over gaps gets one more bin: the wrong answers that no page holds (no clock slips there).
        model_pooled = pool_over_seeds(per_seed, "model")
        landed_by_model = np.append(model_pooled["by_gap"], model_pooled["nowhere"])
        for name in ("model", "designed", "neighbours", "chance"):
            pooled = pool_over_seeds(per_seed, name)
            slips[name] = {
                "neighbour_share": neighbour_share(pooled["by_gap"]),
                "share_landed_nowhere": pooled["nowhere"] / (pooled["nowhere"] + pooled["by_gap"].sum()),
                "distance_from_model": distance(landed_by_model, np.append(pooled["by_gap"], pooled["nowhere"])),
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


def summarise_probes(checked: list) -> dict:
    """Predictions 2 and 3, and Theorem 5 inside the trained models, from the probes of every seed."""
    summary = {"largest_code_cosine": float(np.max([result["largest_code_cosine"] for result in checked]))}
    for n_digits in PROBE_DIGITS:
        summary[n_digits] = summarise_length(n_digits, of_every_seed(checked, "probes", n_digits))

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


# -------------------------------------------------------------------------------------------------------
# The report
# -------------------------------------------------------------------------------------------------------

def interval_text(summary: dict) -> str:
    return f"{summary['mean']:.3f} [{summary['low']:.3f}, {summary['high']:.3f}]"


def signed_interval_text(summary: dict) -> str:
    return f"{summary['mean']:+.3f} [{summary['low']:+.3f}, {summary['high']:+.3f}]"


def print_slips(found: dict):
    for name, slips in found["slips"].items():
        on_a_neighbour = "  -  "                          # no wrong answer landed on a page
        if slips["neighbour_share"] is not None:
            on_a_neighbour = f"{slips['neighbour_share']:.3f}"
        print(f"        {name:10s}  on a neighbour {on_a_neighbour}   "
              f"on no page {slips['share_landed_nowhere']:.3f}   "
              f"distance from the model {slips['distance_from_model']:.3f}")


def print_report(report: dict, n_seeds: int):
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
        print(f"    {n_digits:2d} digits, {found['n_wrong_answers']} wrong answers; "
              f"closest: {found['closest_to_model']}")
        print_slips(found)
    third = probes["prediction_3"]
    print(f"    {third['verdict'].upper()}: {third['rule']}")

    for variant, summary in report["ablations"].items():
        print_ablation(variant, summary, n_seeds)


def print_ablation(variant: str, summary: dict, n_seeds: int):
    lengths = f"{ABLATION_DIGITS[0]}-{ABLATION_DIGITS[-1]} digits"
    print(f"\nAblations of {variant}: accuracy lost when one head is silenced ({lengths}, mean over seeds, "
          f"95% interval)")
    print(f"    with no head silenced, the accuracies equal train.py's on {summary['seeds_matching_train_py']} "
          f"of {n_seeds} seeds")
    for head, lost in summary["heads"].items():
        print(f"    {head:14s}  pure recall {signed_interval_text(lost['pure recall'])}   "
              f"with 1-3 operations {signed_interval_text(lost['with operations'])}")


def main():
    args = parse_arguments()
    seeds = range(args.seeds)

    # The accuracies train.py saved, to check the ablations against. The latest-page diaries are ablated too if every
    # seed has one.
    saved = {"ctm_with_diary": accuracies(load_runs(args.results, "ctm_with_diary", seeds))}
    latest = accuracies(load_runs(args.results, LATEST, seeds))
    if len(latest) == len(seeds):
        saved[LATEST] = latest

    print("probing the trained diaries and silencing their heads")
    jobs = [(args.results, seed, list(saved)) for seed in seeds]
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        checked = pool.map(check_one_seed, jobs, chunksize=1)

    ablations = {}
    for variant in saved:
        ablations[variant] = summarise_heads(of_every_seed(checked, "ablations", variant), saved[variant])
    report = {"probes": summarise_probes(checked), "ablations": ablations}
    print_report(report, args.seeds)

    path = os.path.join(args.results, "probes.json")
    with open(path, "w") as file:
        json.dump(report, file, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
