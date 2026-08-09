# KeenKT Reproduction

This implementation ports KeenKT into the common KT-Toolkit experiment
pipeline.

## Provenance

- Paper: *KeenKT: Knowledge Mastery-State Disambiguation for Knowledge
  Tracing*, AAAI 2026.
- Public source: <https://github.com/HubuKG/KeenKT>
- Audited reference commit:
  `911e4e460b0e2ddb25142fb3c7e46123c1043690`
- Upstream license: Apache-2.0.

The port retains the model's main components:

1. four-parameter Normal-Inverse-Gaussian embeddings;
2. NIG distribution-distance multi-head attention;
3. Rasch-style question-specific variation;
4. diffusion-based denoising reconstruction;
5. distributional contrastive learning;
6. channel recalibration and an MLP response predictor.

## Tensor And Prediction Alignment

For a complete interaction sequence of length `T`:

```text
questions / concepts / responses       [B, T]
strict-causal attention mask           [B, 1, T, T]
full prediction sequence               [B, T]
predictions used by trainer            [B, T-1] = prediction[:, 1:]
shifted response targets               [B, T-1]
select mask                            [B, T-1]
```

At prediction position `t`, the attention mask permits response values only
from positions `< t`. The question and concept at `t` are available because
they describe the target item; the response at `t` is not available.

## Leakage Controls

- The existing fold column determines train and validation users.
- Training uses all configured folds except the current validation fold.
- The labeled public test file is not part of model fitting or checkpoint
  selection.
- Checkpoints are selected by validation AUC only.
- No question-frequency, difficulty, graph, or normalization statistic is
  estimated from validation or test data for KeenKT.
- Response perturbation is constructed only inside training batches and is
  used only by the auxiliary contrastive objective.
- Evaluation does not construct or consume an augmented response sequence.
- Padding positions are excluded from attention keys, pooling, BCE, and
  metrics.
- A causality test changes the current and future responses and verifies that
  the current and earlier predictions are unchanged.

## Deliberate Corrections To The Public Code

The public repository is the architectural reference, but several operations
were adjusted to satisfy the paper's stated objective and causal KT protocol:

- The public SE block averages the complete sequence. This can allow future
  hidden states to influence earlier predictions. The port uses a cumulative
  prefix average.
- Attention uses an explicit strict-past mask and a masked softmax that
  returns zeros when no historical key exists.
- The total objective follows the paper:

  ```text
  BCE + diffusion_weight * diffusion_MSE
      + cl_weight * NIG_contrastive_loss
  ```

  The public trainer applies `cl_weight` a second time to an already weighted
  auxiliary loss.
- Auxiliary pooling divides by the number of valid positions rather than the
  padded sequence length.
- Device placement follows the input tensors and selected toolkit device;
  there is no module-level default CUDA device.

These changes should be disclosed when reporting the reproduction. They make
the implementation leakage-safe, but mean it is not intended to be a
byte-for-byte reproduction of every upstream implementation detail.

## Default Configuration

The `keenkt` block in `configs/kt_config.json` follows the full-model command
in the public repository:

```text
learning_rate       1e-4
weight_decay        1e-5
d_model             256
n_blocks            4
d_ff                512
num_attn_heads      8
dropout             0.2
cl_weight           0.02
diffusion_weight    0.08
noise_level         0.3
emb_type            stoc_qid
```

The AAAI paper separately states an embedding size of 128, batch size 128,
and learning rate `1e-3`. Do not choose between these settings using test
performance. If both protocols are studied, define the alternatives before
training and select hyperparameters using validation results only.

## Commands

One-fold smoke run:

```bash
python scripts/train.py \
  --dataset-name assist2009 \
  --model-name keenkt \
  --fold 0 \
  --num-epochs 1 \
  --use-wandb 0 \
  --save-dir saved_model/smoke
```

Formal five-fold run:

```bash
python scripts/train.py \
  --dataset-name assist2009 \
  --model-name keenkt \
  --cv 1 \
  --folds 0-4 \
  --seed 3407 \
  --use-wandb 0
```

Use the same command with `--dataset-name bridge2algebra2006` for Bridge2006.
