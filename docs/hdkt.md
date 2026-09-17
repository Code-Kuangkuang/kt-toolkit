# HD-KT implementation

The toolkit exposes four HD-KT variants implementing the hybrid
interaction-denoising framework from **HD-KT:
Advancing Robust Knowledge Tracing via Anomalous Learning Interaction
Detection** (WWW 2024).

| Model name | Backbone | Denoised history representation |
| --- | --- | --- |
| `hdkt` (aliases `hd-lpkt`, `hd_lpkt`) | LPKT | exercise/time features before the knowledge-state update |
| `hd_dkt` (alias `hd-dkt`) | DKT | concept-response embeddings before the LSTM |
| `hd_akt` (alias `hd-akt`) | AKT | response-aware attention values; question queries remain unchanged |
| `hd_simplekt` (alias `hd-simplekt`) | SimpleKT | response-aware attention values; question queries remain unchanged |

The `hdkt` variant follows the LPKT backbone exposed by the authors'
`HD-LPKT.py` reference implementation.  The other three variants apply the
same two causal detectors to the repository's existing DKT, AKT and SimpleKT
backbones.

References:

- Paper DOI: <https://doi.org/10.1145/3589334.3645718>
- Reference code: <https://github.com/BIMK/Intelligent-Education/tree/main/HD-KT>

## Model contract

```text
qseqs / shft_qseqs     [B, T-1]
cseqs / shft_cseqs     [B, T-1] or [B, T-1, K]
rseqs / shft_rseqs     [B, T-1]
itseqs / utseqs        [B, T-1]
model predictions      [B, T]
masked prediction      [N_valid]
masked target          [N_valid]
```

The trainer reconstructs full sequences where the backbone requires them.  A
unidirectional GRU and VAE form the knowledge-state detector.  A causal prefix
profile forms the student-profile detector.  Their joint anomaly decision
gates the current interaction before the backbone consumes it.

The public implementation uses a bidirectional sequence encoder and gates the
next exercise representation with a signal computed from the next response.
That can expose future/target information.  This implementation deliberately
uses only the history through `t` when predicting response `t+1`.  It also uses
a history-derived profile rather than a student-ID embedding, which avoids
requiring validation/test student embeddings learned in the training fold.

Padding concepts (`-1`) are masked before embedding.  Multi-concept exercises
are represented with a multi-hot concept association and averaged interaction
embedding.  The time-index vocabulary is fitted from the current fold's
training folds only; unseen validation/test time values use the configured
fallback bucket.

## Training

Single-fold smoke run (replace `hdkt` with any model name in the table):

```bash
python scripts/train.py \
  --dataset-name assist2009 \
  --model-name hdkt \
  --fold 0 \
  --num-epochs 1 \
  --use-wandb 0 \
  --save-dir saved_model/smoke-hdkt
```

Formal five-fold run:

```bash
python scripts/train.py \
  --dataset-name assist2009 \
  --model-name hdkt \
  --cv 1 \
  --folds 0-4 \
  --seed 3407 \
  --use-wandb 0
```

Model selection follows the toolkit default: best validation AUC.  Test
metrics must remain read-only and must not be used to select hyperparameters.
