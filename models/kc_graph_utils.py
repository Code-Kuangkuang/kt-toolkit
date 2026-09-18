"""Question-question adjacency for DenoiseKT, built from the dataset Q-matrix.

Upstream loads this from a `questions_concepts.pt` distributed only through a
Google Drive link, so the file is not reproducible here. It is rebuilt instead,
from `qmatrix.npz`, which the repo already ships per dataset.

What upstream's tensor has to be, read off the two places it is consumed --
`GCN.forward` in models/denoisekt.py and `GCNConv.forward` in pykt's
SFM_CL_model.py, both `torch.sparse.mm(adj, x)` with `x` of shape
`[num_q, emb]` whose rows are then indexed by question id:

  * shape `[num_q, num_q]`. Despite the "questions_concepts" filename it is
    not the `[num_q, num_c]` Q-matrix; that shape would not multiply.
  * already normalised. Neither GCN normalises internally, and the raw binary
    adjacency has degrees up to 1386 on assist2009, which multiplies embedding
    magnitude by ~300 on the first hop.

So this builds `D^-1/2 (A + I) D^-1/2` over `A = 1[Q Qt > 0]`, the standard
Kipf-Welling normalisation, which is also what "gcn_adj" in HCGKT's sibling
filename implies. Measured on assist2009: row sums have median exactly 1.000,
and mean output norm goes from 296 (unnormalised) to 1.02 (normalised) against
an input norm of 16.

That normalisation is an INFERENCE, not something upstream states. It is the
standard choice and the magnitude evidence supports it, but a table that has to
line up with DenoiseKT's published numbers should not assume this file
reproduces the authors' tensor edge for edge.
"""

from __future__ import annotations

import hashlib
import os


def _fingerprint(qmatrix_path: str) -> str:
    """Short digest of the Q-matrix file, so a regenerated dataset misses.

    Same reasoning as models/gkt.py's `_source_fingerprint`: path, size and
    mtime are enough to notice a rebuild without hashing the contents.
    """
    try:
        stat = os.stat(qmatrix_path)
        token = f"{os.path.basename(qmatrix_path)}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        token = f"{qmatrix_path}:missing"
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def build_question_graph(dpath: str, num_q: int):
    """Return the normalised `[num_q, num_q]` adjacency as a sparse tensor.

    Cached next to the data under a fingerprinted name, as GKT's graph is.
    """
    import numpy as np
    import scipy.sparse as sp
    import torch

    qmatrix_path = os.path.join(dpath, "qmatrix.npz")
    if not os.path.exists(qmatrix_path):
        raise FileNotFoundError(
            f"DenoiseKT builds its question graph from {qmatrix_path}, which is "
            f"missing. Regenerate the dataset with scripts/run_clean.py."
        )

    cache = os.path.join(dpath, f"denoisekt_qgraph_{_fingerprint(qmatrix_path)}.npz")
    if os.path.exists(cache):
        cached = np.load(cache)
        indices = torch.tensor(cached["indices"], dtype=torch.long)
        values = torch.tensor(cached["values"], dtype=torch.float32)
        shape = tuple(int(x) for x in cached["shape"])
    else:
        qmatrix = np.load(qmatrix_path)["matrix"]
        # qmatrix carries a trailing padding row (assist2009: 17738 rows for
        # num_q=17737). The graph is over real questions only.
        questions = sp.csr_matrix((qmatrix[:num_q] > 0).astype(np.float32))

        adjacency = questions @ questions.T  # share at least one concept
        adjacency.data[:] = 1.0
        adjacency = adjacency + sp.eye(num_q, format="csr", dtype=np.float32)
        adjacency.data[:] = 1.0  # self-loops, still binary

        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        scale = sp.diags((1.0 / np.sqrt(np.maximum(degree, 1.0))).astype(np.float32))
        normalised = (scale @ adjacency @ scale).tocoo().astype(np.float32)

        index_array = np.vstack([normalised.row, normalised.col])
        np.savez_compressed(
            cache,
            indices=index_array,
            values=normalised.data,
            shape=np.array(normalised.shape),
        )
        indices = torch.tensor(index_array, dtype=torch.long)
        values = torch.tensor(normalised.data, dtype=torch.float32)
        shape = normalised.shape

    return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()


def build_question_concept_map(dpath: str, num_q: int, max_concepts: int):
    """`[num_q, max_concepts]` of concept ids per question, `-1` padded.

    HCGKT loads this as `question_concept_map.npy`, another Google-Drive-only
    file. `get_kc_embedding` in models/sfm_cl.py reads it with
    `padding_idx=-1` and mean-pools each question's concept rows, which fixes
    both the padding value and the shape.
    """
    import numpy as np

    qmatrix_path = os.path.join(dpath, "qmatrix.npz")
    qmatrix = np.load(qmatrix_path)["matrix"][:num_q] > 0

    concept_map = np.full((num_q, max_concepts), -1, dtype=np.int64)
    rows, cols = np.nonzero(qmatrix)
    # np.nonzero yields rows in ascending order, so the running position within
    # each row is just the offset from where that row's block starts.
    starts = np.searchsorted(rows, np.arange(num_q))
    slots = np.arange(len(rows)) - starts[rows]
    keep = slots < max_concepts
    if not keep.all():
        raise ValueError(
            f"{int((~keep).sum())} question-concept pairs do not fit in "
            f"max_concepts={max_concepts}; the Q-matrix has a question with more "
            f"concepts than keyid2idx.json records."
        )
    concept_map[rows[keep], slots[keep]] = cols[keep]
    return concept_map


def load_kc_text_embeddings(dataset_name: str, num_c: int, root_dir: str = "."):
    """BGE embeddings of the concept texts, as `[num_c, dim]`.

    These cannot be derived -- they encode concept *descriptions*, which the
    preprocessed dataset does not carry -- so unlike the graph they are a
    downloaded artefact, kept under utils/kc_embedding/.

    Alignment was checked against assist2009 rather than assumed: the file has
    exactly 123 rows for num_c=123, and 113 of 123 entries in the accompanying
    `kcs_context_assist2009.json` name the same skill, by index, as this repo's
    `keyid2idx.json` does. The remaining 10 are skills the source CSV leaves
    unnamed, where the authors substituted a random placeholder string. The
    authors' notebook also indexes concepts straight off pykt's
    `*_quelevel.csv`, which is the same index space this repo uses.

    The row-count check below is what protects a different dataset from being
    wired up on the assumption that the same holds there.
    """
    import numpy as np

    path = os.path.join(
        root_dir, "utils", "kc_embedding", f"kc_embeddings_{dataset_name}_bge.npy"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HCGKT needs concept-text embeddings at {path}. They are not "
            f"derivable from the preprocessed data and are the paper authors' "
            f"files, so they are gitignored rather than redistributed here. "
            f"Download kc_embeddings_<dataset>_bge.npy from the folder HCGKT's "
            f"own source points at: "
            f"https://drive.google.com/drive/folders/1cUqLbBRlj_PPIIhySghyaIjlasIDGIwF "
            f"and put it in utils/kc_embedding/. The kcs_context_<dataset>.json "
            f"files already in that directory are the matching concept texts."
        )
    embeddings = np.load(path)
    if embeddings.shape[0] != num_c:
        raise ValueError(
            f"{path} has {embeddings.shape[0]} concept rows but {dataset_name} "
            f"has num_c={num_c}. The concept indexing differs, so these vectors "
            f"would attach to the wrong concepts silently."
        )
    return embeddings
