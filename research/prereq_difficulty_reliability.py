"""Does borrowing difficulty along junyi's prerequisite edges beat just counting?

No model, no GPU, no training. The question is a property of the data: when an
item has too few observations to estimate its difficulty, does shrinking that
estimate towards its prerequisites' difficulty produce a more *reliable* number
than shrinking it towards the global mean?

Measured by split-half reliability. For each item we draw 2n of its observations,
split them into two halves of n, estimate difficulty independently in each half,
and correlate the two estimates across items (Spearman). A difficulty estimate
that cannot reproduce itself on a second sample of the same size cannot be
carrying item information, whatever a downstream model does with it.

Four estimators, and the choice of the last two is the whole point:

  plain    log-odds with a Haldane 0.5 correction. Nothing borrowed.
  global   shrink towards the global log-odds, weight alpha.
           *This is the null*, not `plain`. Shrinking towards the grand mean
           already buys reliability at small n for free and without any
           structure, so an estimator that only beats `plain` has shown nothing.
  prereq   shrink towards the mean of the item's prerequisites' plain estimates,
           computed inside the same half.
  random   identical to `prereq` but on a degree-preserving random rewiring of
           the prerequisite graph. Separates "this structure" from "any
           smoothing over a graph of this shape". If prereq ~= random, the
           curriculum annotation is decoration.

Prediction being tested, stated before running: prereq > global at small n,
converging as n grows, and prereq > random everywhere. Any other shape is a
negative result for prerequisite-structured difficulty correction.

Usage:
    python research/prereq_difficulty_reliability.py
    python research/prereq_difficulty_reliability.py --max-rows 20000 --repeats 3
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DPATH = ROOT / "data" / "junyi2015"
csv.field_size_limit(10 ** 9)


def load_prereq_graph():
    """exercise-table prerequisites, mapped into the repo's question index space.

    `keyid2idx.json` stores question names with '_' replaced by '####'.
    """
    key2idx = json.loads((DPATH / "keyid2idx.json").read_text(encoding="utf-8"))["questions"]
    norm = lambda s: s.strip().replace("_", "####")
    parents = defaultdict(list)
    topic_of = {}
    with open(DPATH / "junyi_Exercise_table.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            child = key2idx.get(norm(row["name"]))
            if child is None:
                continue
            topic_of[child] = row["topic"]
            for p in row["prerequisites"].split(","):
                pid = key2idx.get(norm(p))
                if pid is not None and pid != child:
                    parents[child].append(pid)
    by_topic = defaultdict(list)
    for item, t in topic_of.items():
        by_topic[t].append(item)
    topic_groups = {item: by_topic[t] for item, t in topic_of.items()}
    return dict(parents), topic_groups, len(key2idx)


def repartition(groups, rng):
    """Random partition with the identical multiset of group sizes."""
    sizes = [len(v) for v in {id(v): v for v in groups.values()}.values()]
    members = list(groups)
    rng.shuffle(members)
    out, i = {}, 0
    for s in sizes:
        chunk = members[i:i + s]
        i += s
        for m in chunk:
            out[m] = chunk
    for m in members[i:]:
        out[m] = [m]
    return out


def rewire(parents, num_q, rng):
    """Degree-preserving random rewiring: same in-degree per node, random targets."""
    out = {}
    for child, ps in parents.items():
        pool = [q for q in range(num_q) if q != child]
        out[child] = rng.sample(pool, min(len(ps), len(pool)))
    return out


def load_observations(folds, max_rows):
    """(item -> list of 0/1 responses) from the quelevel sequences.

    All five folds by default. The usual `feature_fit_scope: train_folds`
    discipline exists to stop a statistic derived from held-out rows leaking
    into an evaluation on those rows; here nothing is fitted and nothing is
    evaluated out of sample -- the split being measured is the split-half
    *inside* each item's own observations. Restricting to training folds would
    only shrink n without removing any leak.

    `max_rows` counts rows *kept*, not rows scanned: the file is sorted by fold,
    so capping the scan silently returned zero rows when folds were filtered.
    """
    obs = defaultdict(list)
    path = DPATH / "train_valid_sequences_quelevel.csv"
    kept = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["fold"]) not in folds:
                continue
            qs = row["questions"].split(",")
            rs = row["responses"].split(",")
            for q, r in zip(qs, rs):
                if q == "-1" or r == "-1":
                    continue
                obs[int(q)].append(int(r))
            kept += 1
            if max_rows and kept >= max_rows:
                break
    return obs


def plain_logodds(correct, total):
    """Haldane-corrected log-odds. Defined at total=0 and at p_hat in {0, 1}."""
    return math.log((correct + 0.5) / (total - correct + 0.5))


def estimate_half(sample, parents, alpha, global_lo):
    """All four estimators for one half. `sample` is item -> list of responses."""
    plain, n_obs = {}, {}
    for item, rs in sample.items():
        plain[item] = plain_logodds(sum(rs), len(rs))
        n_obs[item] = len(rs)

    def shrink_towards(target_of):
        out = {}
        for item in plain:
            t = target_of(item)
            if t is None:
                t = global_lo
            n = n_obs[item]
            out[item] = (n * plain[item] + alpha * t) / (n + alpha)
        return out

    def parent_mean(graph):
        def f(item):
            ps = [plain[p] for p in graph.get(item, []) if p in plain]
            return sum(ps) / len(ps) if ps else None
        return f

    def group_mean(groups):
        """Mean over the item's whole group, excluding itself.

        Self-exclusion matters: including the item would leak its own noisy
        estimate into its own shrinkage target, which inflates split-half
        agreement without any information having been borrowed.
        """
        def f(item):
            peers = [plain[p] for p in groups.get(item, ()) if p != item and p in plain]
            return sum(peers) / len(peers) if peers else None
        return f

    return {
        "plain": plain,
        "global": shrink_towards(lambda _item: global_lo),
        "prereq": shrink_towards(parent_mean(parents["real"])),
        "random": shrink_towards(parent_mean(parents["rand"])),
        "topic": shrink_towards(group_mean(parents["topic"])),
        "toprand": shrink_towards(group_mean(parents["toprand"])),
    }


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[3, 5, 10, 30, 100])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-rows", type=int, default=60000)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    real_parents, topic_groups, num_q = load_prereq_graph()
    print(f"prerequisite graph: {len(real_parents)} items with parents, "
          f"{sum(len(v) for v in real_parents.values())} edges, over {num_q} questions")

    print(f"loading observations from folds {args.folds} (max_rows={args.max_rows}) ...",
          flush=True)
    obs = load_observations(set(args.folds), args.max_rows)
    total = sum(len(v) for v in obs.values())
    print(f"  {total:,} interactions over {len(obs)} items "
          f"({total / max(len(obs), 1):,.0f} per item)\n")

    all_c = sum(sum(v) for v in obs.values())
    global_lo = plain_logodds(all_c, total)
    print(f"global log-odds = {global_lo:.4f}  (p = {all_c / total:.4f})\n")

    names = ["plain", "global", "prereq", "random", "topic", "toprand"]
    print(f"{'target':>7}{'med n':>7}" + "".join(f"{x:>8}" for x in names)
          + f"{'preq-rand':>15}{'topic-rand':>16}")
    rows_out = []
    for n in args.ns:
        eligible = [i for i, v in obs.items() if len(v) >= 2 * n]
        if len(eligible) < 20:
            print(f"{n:>7}{len(eligible):>7}   too few items, skipped")
            continue
        acc = defaultdict(list)
        achieved = []
        for rep in range(args.repeats):
            r2 = random.Random(args.seed + rep)
            parents = {"real": real_parents,
                       "rand": rewire(real_parents, num_q, r2),
                       "topic": topic_groups,
                       "toprand": repartition(topic_groups, r2)}
            halves = ({}, {})
            # Thin every item by the SAME rate rather than giving every item the
            # same count. Equal counts make the `global` estimator a linear
            # transform of `plain` -- identical n, identical alpha, constant
            # target -- and Spearman is invariant to that, so the null came out
            # numerically equal to the strawman and measured nothing. Real item
            # frequencies are heterogeneous, and shrinking a rarely-seen item
            # harder than a common one is exactly how the grand mean earns its
            # reliability. Preserving that spread is what makes `global` a null.
            rate = (2 * n) / (sum(len(v) for v in obs.values()) / len(obs))
            for item in eligible:
                k = int(round(len(obs[item]) * rate))
                if k < 2:
                    continue
                draw = r2.sample(obs[item], min(k, len(obs[item])))
                h = len(draw) // 2
                halves[0][item] = draw[:h]
                halves[1][item] = draw[h:2 * h]
            achieved.append(
                sorted(len(v) for v in halves[0].values())[len(halves[0]) // 2])
            est = [estimate_half(h, parents, args.alpha, global_lo) for h in halves]
            for name in names:
                a, b = est[0][name], est[1][name]
                keys = sorted(set(a) & set(b))
                acc[name].append(spearman([a[k] for k in keys], [b[k] for k in keys]))
        means = {k: sum(v) / len(v) for k, v in acc.items()}
        med = sum(achieved) / len(achieved)

        def sd(v):
            if len(v) < 2:
                return 0.0
            m = sum(v) / len(v)
            return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

        # Paired across repeats: the same resample feeds both estimators, so the
        # sd of the per-repeat difference is the number that says whether the
        # gap survives resampling noise. The sd of each mean separately does not.
        pairs = [p - r for p, r in zip(acc["prereq"], acc["random"])]
        tpairs = [p - r for p, r in zip(acc["topic"], acc["toprand"])]
        d_pr = sum(pairs) / len(pairs)
        d_tp = sum(tpairs) / len(tpairs)
        print(f"{n:>7}{med:>7.0f}"
              + "".join(f"{means[x]:>8.4f}" for x in names)
              + f"{d_pr:>+9.4f}±{sd(pairs):.3f}"
              + f"{d_tp:>+9.4f}±{sd(tpairs):.3f}")
        rows_out.append({"n_target": n, "median_n_per_half": med,
                         "items": len(eligible),
                         **{k: [round(sum(v) / len(v), 5), round(sd(v), 5)]
                            for k, v in acc.items()},
                         "prereq_minus_random": [round(d_pr, 5), round(sd(pairs), 5)],
                         "topic_minus_toprand": [round(d_tp, 5), round(sd(tpairs), 5)]})

    out = ROOT / "experiment" / "prereq_difficulty_reliability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "rows": rows_out},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out}")
    print("\nreading: `global` is the null. `prereq` must beat it, and must beat "
          "`random`, or prerequisite structure adds nothing over generic shrinkage.")


if __name__ == "__main__":
    main()
