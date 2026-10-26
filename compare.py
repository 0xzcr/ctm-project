"""
Compare the accuracies that train.py saved. No model is loaded, so it takes a few seconds.

    python compare.py

It needs ctm and ctm_with_diary trained for seeds 0-19 (see the README), and checks three of the thesis's predictions
(its section "What to test next"):

    1. The two-digit drop shrinks   two-digit accuracy well above Box 1's 0.47-0.67      against: staying in that range
    4. Length                       recall with 5-16 digits stays close to recall with 1-4   against: a fall like LEM's
    5. Every seed reported

Then it compares every extra variant that has been trained with its baseline, paired by seed:

    size          the diary against a plain CTM with 107 neurons, which has as many parameters    (ctm_d107)
    latest page   the diary with a fifth head that reads the latest page, against the diary    (ctm_with_diary_latest)
    images        the diary against the plain CTM, both shown real MNIST images      (ctm_with_diary_mnist, ctm_mnist)
                  from scratch, and with the image backbone first trained to tell digits apart and then frozen,
                  trained 4x longer on 5 seeds                                     (..._mnist_pretrained_steps16000)

The report is printed and saved in results/comparisons.json. probe.py checks predictions 2 and 3 inside the models.
"""
import argparse
import itertools
import json
import os

import numpy as np
from scipy import stats

from train import LONGER_DIGITS, RESULTS_FOLDER, TRAIN_DIGITS, TRAIN_OPS

ALL_DIGITS = TRAIN_DIGITS + LONGER_DIGITS
EARLIER_RESULTS = os.path.expanduser("~/ctm/results/primary")   # the runs behind Box 1; checks that need them are
                                                                # skipped where they are not
BOX1_BEST_TWO_DIGITS = 0.668          # the best two-digit accuracy in Box 1 (the CTM with a long history)
CLOSE_ENOUGH = 0.9                    # prediction 4 holds if 5-16 digits keep at least 90% of the 1-4 accuracy
SEED_TABLE_DIGITS = (1, 2, 4, 8, 16)  # the lengths prediction 5 reports for every seed

EXTRA_VARIANTS = ("ctm_d107", "ctm_with_diary_latest", "ctm_mnist", "ctm_with_diary_mnist",
                  "ctm_mnist_pretrained_steps16000", "ctm_with_diary_mnist_pretrained_steps16000")

# (name, variant, baseline, the question the pair answers)
COMPARISONS = (
    ("size", "ctm_with_diary", "ctm_d107", "is the diary's gain only its extra parameters?"),
    ("latest page", "ctm_with_diary_latest", "ctm_with_diary",
     "were the errors left over from having nowhere to keep the running result?"),
    ("images, from scratch", "ctm_with_diary_mnist", "ctm_mnist",
     "does the diary still help when the digits are real MNIST images?"),
    ("images, pretrained backbone", "ctm_with_diary_mnist_pretrained_steps16000", "ctm_mnist_pretrained_steps16000",
     "does it help once the frozen image backbone can tell the digits apart, with 4x the training?"),
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=RESULTS_FOLDER, help="where train.py saved")
    parser.add_argument("--seeds", type=int, default=20, help="use seeds 0 to this number minus one")
    return parser.parse_args()


# -------------------------------------------------------------------------------------------------------
# Loading, and the numbers every check uses
# -------------------------------------------------------------------------------------------------------

def load_runs(folder: str, variant: str, seeds) -> dict:
    """{seed: what train.py saved} for the seeds of a variant that have been trained, maybe none."""
    runs = {}
    for seed in seeds:
        path = os.path.join(folder, f"{variant}_seed{seed}.json")
        if os.path.exists(path):
            with open(path) as file:
                runs[seed] = json.load(file)
    return runs


def accuracies(runs: dict) -> dict:
    """{seed: the accuracy report}: {n_digits: {"mean": accuracy over 0-3 operations, "by_ops": {n_ops: accuracy}}}."""
    reports = {}
    for seed, run in runs.items():
        reports[seed] = run["accuracy"]
    return reports


def cells(report: dict) -> dict:
    """{"n_digits,n_ops": accuracy} for every cell of one report."""
    flat = {}
    for n_digits, n_ops in itertools.product(ALL_DIGITS, TRAIN_OPS):
        flat[f"{n_digits},{n_ops}"] = report[str(n_digits)]["by_ops"][str(n_ops)]
    return flat


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


def mean_with_ops(report: dict, n_ops: int) -> float:
    """One seed's accuracy with n_ops operations, averaged over every number of digits evaluated."""
    values = []
    for n_digits in ALL_DIGITS:
        values.append(report[str(n_digits)]["by_ops"][str(n_ops)])
    return np.mean(values)


def accuracy_with_ops(reports: dict, n_ops: int) -> np.ndarray:
    """(seeds,) mean_with_ops() of every seed."""
    values = []
    for report in reports.values():
        values.append(mean_with_ops(report, n_ops))
    return np.array(values)


def mean_with_interval(values: np.ndarray) -> dict:
    """The mean over seeds and its 95% confidence interval (Student t)."""
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "low": mean, "high": mean}
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": mean, "low": float(mean - half_width), "high": float(mean + half_width)}


def kept(reports: dict) -> float:
    """The share of its 1-4 digit accuracy that a variant keeps at 5-16 digits (means over seeds)."""
    return float(mean_over_lengths(reports, LONGER_DIGITS).mean() / mean_over_lengths(reports, TRAIN_DIGITS).mean())


# -------------------------------------------------------------------------------------------------------
# The thesis's predictions 1, 4 and 5
# -------------------------------------------------------------------------------------------------------

def same_as_box1(plain: dict) -> dict:
    """For how many seeds every accuracy cell equals the earlier run behind Box 1 (if those runs are here)."""
    if not os.path.isdir(EARLIER_RESULTS):
        return {"checked": 0, "identical": 0}

    identical = 0
    for seed, report in plain.items():
        with open(os.path.join(EARLIER_RESULTS, f"ctm_s{seed}.json")) as file:
            earlier_grid = json.load(file)["grid"]
        identical += all_cells_equal(report, earlier_grid)
    return {"checked": len(plain), "identical": identical}


def all_cells_equal(report: dict, earlier_grid: dict) -> bool:
    """True if every (digits, operations) accuracy in a train.py report equals the earlier run's."""
    for cell, accuracy in cells(report).items():
        if accuracy != earlier_grid[cell]["acc"]:
            return False
    return True


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


def lem_length_kept():
    """The share of its 1-4 digit accuracy that LEM (log focus) kept at 5-16 digits, from the earlier runs."""
    if not os.path.isdir(EARLIER_RESULTS):
        return None

    shorts = []
    longs = []
    for seed in range(20):
        with open(os.path.join(EARLIER_RESULTS, f"mem_log_s{seed}.json")) as file:
            curve = json.load(file)["recall_curve"]
        shorts.append(curve_mean(curve, TRAIN_DIGITS))
        longs.append(curve_mean(curve, LONGER_DIGITS))
    return float(np.mean(longs) / np.mean(shorts))


def curve_mean(curve: dict, lengths) -> float:
    """The average of an earlier run's recall curve over several numbers of digits."""
    values = []
    for n_digits in lengths:
        values.append(curve[str(n_digits)])
    return np.mean(values)


def accuracy_by_operations(reports: dict) -> dict:
    """{n_ops: {n_digits: accuracy, averaged over seeds}}. Shows which questions the errors come from."""
    table = {}
    for n_ops in range(4):
        table[n_ops] = {}
    for n_ops, n_digits in itertools.product(range(4), ALL_DIGITS):
        table[n_ops][n_digits] = float(np.mean(cell_by_seed(reports, n_digits, n_ops)))
    return table


def cell_by_seed(reports: dict, n_digits: int, n_ops: int) -> list:
    """Every seed's accuracy with n_digits digits and n_ops operations."""
    values = []
    for report in reports.values():
        values.append(report[str(n_digits)]["by_ops"][str(n_ops)])
    return values


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

    result["by_operations"] = {"ctm": accuracy_by_operations(plain), "ctm_with_diary": accuracy_by_operations(diary)}
    result["lem_kept"] = lem_kept
    result["verdict"] = verdict
    result["rule"] = (f"holds if 5-16 digits keep at least {CLOSE_ENOUGH:.0%} of the 1-4 accuracy; "
                      "fails if they keep no more than LEM did")
    return result


def every_seed(plain: dict, diary: dict) -> list:
    """Prediction 5: every seed's accuracy, both variants, at a few lengths."""
    rows = []
    for seed in plain:
        rows.append(seed_row(seed, plain[seed], diary[seed]))
    return rows


def seed_row(seed: int, plain_report: dict, diary_report: dict) -> dict:
    """One seed's row of prediction 5: {"seed", "ctm n", "diary n"} for n = 1, 2, 4, 8 and 16 digits."""
    row = {"seed": seed}
    for n_digits in SEED_TABLE_DIGITS:
        row[f"ctm {n_digits}"] = plain_report[str(n_digits)]["mean"]
        row[f"diary {n_digits}"] = diary_report[str(n_digits)]["mean"]
    return row


# -------------------------------------------------------------------------------------------------------
# The extra variants, each against its baseline
# -------------------------------------------------------------------------------------------------------

def mean_by_length(reports: dict) -> dict:
    """{n_digits: accuracy averaged over 0-3 operations and seeds}, and the number of seeds."""
    row = {"seeds": len(reports)}
    for n_digits in ALL_DIGITS:
        row[n_digits] = float(accuracy_by_seed(reports, n_digits).mean())
    return row


def backbone_test_accuracy(runs: dict):
    """The mean over seeds of the pretrained backbone's accuracy on MNIST test digits, or None without pretraining."""
    values = []
    for run in runs.values():
        value = run.get("pretrained_backbone_test_accuracy")
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def compare(variant: dict, baseline: dict) -> dict:
    """
    A variant against its baseline, paired by seed: the difference in accuracy (variant minus baseline) with two
    digits, with 1-4 and with 5-16 digits, and with each number of operations; on how many seeds the variant is better
    over all lengths; and how much of its 1-4 digit accuracy each one keeps at 5-16 digits.
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


def comparisons(runs: dict) -> dict:
    """compare() for every pair in COMPARISONS whose two variants have been trained, on the seeds both have."""
    result = {}
    for name, variant, baseline, question in COMPARISONS:
        if variant not in runs or baseline not in runs:
            continue
        variant_runs, baseline_runs = paired_runs(runs[variant], runs[baseline])
        if not variant_runs:
            continue
        first_seed = list(variant_runs)[0]
        result[name] = {
            "variant": variant,
            "baseline": baseline,
            "question": question,
            "seeds": len(variant_runs),
            "parameters": {"variant": variant_runs[first_seed]["parameters"],
                           "baseline": baseline_runs[first_seed]["parameters"]},
            "backbone_test_accuracy": {"variant": backbone_test_accuracy(variant_runs),
                                       "baseline": backbone_test_accuracy(baseline_runs)},
            **compare(accuracies(variant_runs), accuracies(baseline_runs)),
        }
    return result


def paired_runs(variant_runs: dict, baseline_runs: dict) -> tuple[dict, dict]:
    """The runs of both variants on the seeds both have, in the same order, so the two pair up seed by seed."""
    variant_paired = {}
    baseline_paired = {}
    for seed, run in variant_runs.items():
        if seed in baseline_runs:
            variant_paired[seed] = run
            baseline_paired[seed] = baseline_runs[seed]
    return variant_paired, baseline_paired


# -------------------------------------------------------------------------------------------------------
# The report
# -------------------------------------------------------------------------------------------------------

def interval_text(summary: dict) -> str:
    return f"{summary['mean']:.3f} [{summary['low']:.3f}, {summary['high']:.3f}]"


def signed_interval_text(summary: dict) -> str:
    return f"{summary['mean']:+.3f} [{summary['low']:+.3f}, {summary['high']:+.3f}]"


def accuracies_text(row: dict) -> str:
    """One accuracy per number of digits, in columns 7 characters wide."""
    text = ""
    for n_digits in ALL_DIGITS:
        text += f"{row[n_digits]:7.3f}"
    return text


def print_operations_table(name: str, table: dict):
    for n_ops, row in table.items():
        print(f"    {name:15s} {n_ops} ops" + accuracies_text(row))


def print_seed_row(row: dict):
    line = f"    {row['seed']:4d}"
    for n_digits in SEED_TABLE_DIGITS:
        line += f"{row[f'ctm {n_digits}']:>10.3f}{row[f'diary {n_digits}']:>10.3f}"
    print(line)


def print_comparison(name: str, pair: dict):
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


def print_report(report: dict):
    box1 = report["same_as_box1"]
    print(f"\nThe plain CTM here is Box 1's CTM: {box1['identical']} of {box1['checked']} seeds identical "
          "in every cell")

    first = report["prediction_1"]
    print("\nPrediction 1, the two-digit drop shrinks (mean over seeds, 95% interval)")
    print(f"    CTM                    {interval_text(first['ctm'])}")
    print(f"    CTM with the diary     {interval_text(first['ctm_with_diary'])}")
    print(f"    difference, paired     {interval_text(first['difference'])}")
    print(f"    {'HOLDS' if first['holds'] else 'DOES NOT HOLD'}: {first['rule']}")

    fourth = report["prediction_4"]
    print("\nPrediction 4, length (mean accuracy over 0-3 operations and seeds)")
    for name in ("ctm", "ctm_with_diary"):
        row = fourth[name]
        print(f"    {name:15s} 1-4 digits {row['1-4 digits']:.3f}   5-16 digits {row['5-16 digits']:.3f}   "
              f"kept {row['kept']:.0%}")
    if fourth["lem_kept"] is not None:
        print(f"    LEM (log focus) kept {fourth['lem_kept']:.0%}")
    print(f"    {fourth['verdict'].upper()}: {fourth['rule']}")
    print("    accuracy by number of operations (mean over seeds), with 1 to 16 digits:")
    print("                      " + "".join(f"{n_digits:>7d}" for n_digits in ALL_DIGITS))
    for name, table in fourth["by_operations"].items():
        print_operations_table(name, table)

    print("\nPrediction 5, every seed (mean accuracy over 0-3 operations; ctm n / diary n = with n digits)")
    header = "    seed"
    for n_digits in SEED_TABLE_DIGITS:
        header += f"{'ctm ' + str(n_digits):>10}{'diary ' + str(n_digits):>10}"
    print(header)
    for row in report["prediction_5"]:
        print_seed_row(row)

    print("\nEvery variant: mean accuracy over 0-3 operations and seeds, by number of digits")
    print(" " * 50 + "".join(f"{n_digits:>7d}" for n_digits in ALL_DIGITS))
    for name, row in report["accuracy_by_length"].items():
        print(f"    {name:42s}{row['seeds']:2d} seeds" + accuracies_text(row))

    for name, pair in report["comparisons"].items():
        print_comparison(name, pair)


def main():
    args = parse_arguments()
    seeds = range(args.seeds)

    runs = {}
    for variant in ("ctm", "ctm_with_diary") + EXTRA_VARIANTS:
        found = load_runs(args.results, variant, seeds)
        if found:
            runs[variant] = found
    for variant in ("ctm", "ctm_with_diary"):
        if len(runs.get(variant, {})) < args.seeds:
            raise SystemExit(f"{variant} needs seeds 0-{args.seeds - 1} trained in {args.results} (see the README)")

    plain = accuracies(runs["ctm"])
    diary = accuracies(runs["ctm_with_diary"])
    by_length = {}
    for variant, variant_runs in runs.items():
        by_length[variant] = mean_by_length(accuracies(variant_runs))

    report = {
        "same_as_box1": same_as_box1(plain),
        "prediction_1": two_digit_drop(plain, diary),
        "prediction_4": length(plain, diary),
        "prediction_5": every_seed(plain, diary),
        "accuracy_by_length": by_length,
        "comparisons": comparisons(runs),
    }
    print_report(report)

    path = os.path.join(args.results, "comparisons.json")
    with open(path, "w") as file:
        json.dump(report, file, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
