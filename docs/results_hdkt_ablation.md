# HD-KT ablation, assist2009, clean labels

Run 2026-09-18. Recorded here because `saved_model/` and `experiment/` are both
gitignored, so the artifacts do not survive a clean and the numbers cost 155
minutes of GPU.

This is the **noise ratio = 0 anchor**, not the experiment. HD-KT is a
robustness method: its claim is that accuracy degrades more slowly as label
noise rises, not that it is more accurate on clean data. The noise sweep
(`--train_label_flip_ratio`) has not been run.

## Protocol

All 30 runs share one protocol block, checked by
`scripts/run_baseline_table.py::protocol_key`:

```json
{"dataset_mode": "all_in_one", "concept_mode": "multi", "max_concepts": 4,
 "concepts_visible": "all", "score_repeated_kc": false, "eval_window": true,
 "feature_fit_scope": "none", "graph_scope": "none"}
```

seed 3407, folds 0-4, `saved_model/baseline_table`.

This is the first table in this repository whose rows are all under one
protocol; earlier stored baselines predate the `is_repeat` scoring fix and the
protocol stamp, and are not comparable to it.

## Result

Windowed test AUC, paired over the five folds.

| pair | base | HD | delta | sd(delta) | t (df=4) | folds won |
|---|---|---|---|---|---|---|
| `dkt` → `hd_dkt` | 0.76069 | 0.76063 | −0.00006 | 0.00128 | −0.11 | 2/5 |
| `akt` → `hd_akt` | 0.78100 | 0.78095 | −0.00004 | 0.00305 | −0.03 | 2/5 |
| `simplekt` → `hd_simplekt` | 0.78403 | 0.78522 | +0.00120 | 0.00314 | +0.85 | 3/5 |

Per-fold deltas:

```
hd_dkt        -0.00040  +0.00047  -0.00068  -0.00153  +0.00184
hd_akt        -0.00340  +0.00462  +0.00094  -0.00175  -0.00063
hd_simplekt   +0.00249  -0.00240  -0.00193  +0.00434  +0.00348
```

|t| would need ~2.78 for p < 0.05 at df = 4. None of the three is close, and
every pair changes sign across folds.

**Reading:** on clean data the denoiser does nothing -- which is what it should
do. The useful content is the absence of harm: a gate that mis-fires on clean
interactions would show up here as a consistent negative, and it does not. That
is the precondition for the noise-slope experiment being interpretable at all,
and it now holds.

## A capacity difference the table does not show

Found on 2026-09-18 by `core/model_info.py`, after these runs finished. On
assist2009 (`num_q` = 17,737):

| pair | backbone | composed | composed / backbone |
|---|---|---|---|
| `dkt` → `hd_dkt` | 395,523 | 4,584,255 | **11.59x** |
| `akt` → `hd_akt` | 1,528,931 | 7,012,327 | **4.59x** |
| `simplekt` → `hd_simplekt` | 5,773,313 | 11,256,709 | **1.95x** |

Almost all of it is one tensor: the denoiser's own question-embedding table,
`plugin.denoiser.item_embed.weight`, which is `num_q x d` and alone accounts for
3.5M of HD-DKT's 4.6M parameters. It is separate from the backbone's own
embedding, so a plugged model carries two.

So these rows are not "backbone vs backbone + denoising". They are "backbone vs
a 2-12x larger model that also denoises", and a reader will ask about it.

Two things to say about it honestly:

- The null result is not explained by under-capacity. Every HD model is strictly
  larger and none of them wins.
- The ordering runs the *wrong* way for a capacity story: the pair with the
  least extra capacity (`simplekt`, 1.95x) is the only one with a positive
  delta, and the pair with the most (`dkt`, 11.59x) is flat. If extra
  parameters were driving anything here, it is not visible.

A cleanly separated experiment would size the denoiser's item embedding to
match, or share the backbone's -- `models/hdkt.py` already does the latter, via
`exercise_projection` over the backbone's `e_embed`, while
`modules/hd_denoiser.py` keeps its own table. That divergence between the two
HD implementations is untouched.

## What makes this comparison valid

Three fixes landed the same day, each of which had been a confound:

| fix | commit | had been |
|---|---|---|
| composed HD models share their backbone's trainer | `7d77c2a` | HD trainers used float32 BCE against baselines' float64; HD-SimpleKT dropped SimpleKT's item L2 penalty |
| `hdkt` matched to LPKT's concept weighting | `d2dc92a` | different `gamma` smoothing and one fewer knowledge slot |
| `AKTTrainer` tolerates a missing `qseqs` | `7d77c2a` | `akt` could not run on concept-only datasets |

`hdkt` / `lpkt` are **not** in this table: they are configured with different
learning rates (0.001 vs 0.003), which is a second difference in any row
comparing them. See `docs/hdkt.md`.

## Cost, for planning the noise sweep

| model | min/run |
|---|---|
| `dkt`, `hd_dkt`, `simplekt`, `hd_simplekt` | 0.6 – 1.2 |
| `akt` | 11.6 |
| `hd_akt` | 16.0 |

The AKT pair is 89% of the wall clock. A noise sweep over the two cheap pairs at
three ratios is about an hour; adding AKT makes it eight. Run the cheap pairs
first and only extend if a slope appears.

Use a separate `--save-root` per ratio: `run_dir_name` does not include the flip
ratio, so a second ratio under the same root is either skipped as already
complete or silently mixed with the first.
