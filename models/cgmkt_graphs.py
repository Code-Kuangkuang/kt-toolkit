"""Graph and group artefacts for CGMKT, with the paper/code split made explicit.

CGMKT's paper and its released code disagree in two places, and both disagreements
sit on components the paper names in its title. Rather than pick one and hide the
choice, every artefact here is selectable, so a run states which variant it used
and the two can be compared on the same folds.

Disagreement 1 -- the question graph.
    The paper (Sec. 4.1.2) says the GCN operates on
    `Aq = D^-1/2 (A At) D^-1/2`, a question-question co-occurrence graph, and
    states three separate times that the incidence matrix `A` is NOT what the
    GCN sees ("Importantly, A is an incidence matrix, not the adjacency matrix
    supplied to the GCN").
    The released `ques_skill_gcn_adj.pt` is `[num_q, num_q]` with non-zero
    columns only in `0..num_c-1` -- measured on assist2009: shape (17737, 17737),
    non-zero columns 0..122 for num_c=123, asymmetric. It is the Q-matrix
    zero-padded to square and row-normalised, i.e. exactly the incidence matrix
    the paper says is not used. `torch.sparse.mm(A_padded, H_q)` then mean-pools
    the first num_c rows of the question embedding table into each question, so
    the component is a Q-matrix KC aggregation, not a question graph.
    This matters because the ablation makes the question branch the only
    component carrying accuracy (-0.0259 AUC on ASSIST2009, against <=0.003 for
    everything else).
    `question_graph_source="incidence"` reproduces the code; `"cooccur"` builds
    what the paper describes, reusing `kc_graph_utils.build_question_graph`.

Disagreement 2 -- the KC graph and the groups.
    The paper says the KC dependency graph is "learned by a stochastic block
    model". The authors' `cgmkt_preprocess.ipynb` generates it by instantiating
    a fresh `SBMAttention` per k, running ONE forward pass over the BGE concept
    embeddings and Bernoulli-sampling the result. There is no optimiser, no
    loss and no likelihood anywhere in that cell, and the notebook's own comment
    records that the seed was not fixed ("SBM 图生成随机, 原脚本未固定种子").
    Nothing is fitted to data, so "learned" describes the algebraic form only.
    Measured on the released assist2009 files (n=123): the graphs have modularity
    0.057-0.074 against 0.063-0.075 for an Erdos-Renyi graph of matched density,
    i.e. no excess structure; and the k=2 and k=9 graphs agree element-wise at
    0.6164 where two independent random draws would agree at 0.6191, i.e. k does
    not change the graph. The mechanism is that `softmax` runs over all k^2
    entries of the block matrix at once, so with Kaiming-random cluster vectors
    in 1024 dimensions the diagonal is only 1.24x the off-diagonal at k=9.
    `kc_graph_source="sbm_random"` reproduces that; `"sbm_fit"` fits a real SBM
    by likelihood to an observed KC graph; `"random"` and `"cooccur"` are
    controls.

    That the two differ is measured, not assumed. Taking the block matrix's
    diagonal/off-diagonal ratio -- the "intra-group cohesion vs cross-group
    interaction" the paper says B encodes -- on assist2009 at k=9:

        sbm_random (upstream pipeline)      0.449 / 0.245 = 1.83x
        sbm_fit    (above-chance transition) 0.594 / 0.133 = 4.46x

    The port of `sbm_random` also lands where the authors' own released files
    do: density 0.263 against their 0.255, spectral-partition modularity 0.058
    against their 0.060 and the paper's reported 0.062.

    `"cooccur"` is degenerate on assist2009 and is kept only as a control. The
    average question carries 1.20 KCs, so the co-occurrence graph has density
    0.0086 and falls apart into near-isolated components; a fit against it
    returns off-diagonal blocks of exactly 0, and `group_transition` then has
    no signal to propagate and falls back to uniform. That is a property of the
    dataset's Q-matrix, not of the construction.

    The membership matrix is a separate axis. The released artefacts contain no
    membership or block file under any of the sixteen names `cgmkt.py` looks
    for, so upstream always falls through to its spectral-clustering fallback --
    silently, since the miss is only recorded in a local string. `group_source`
    makes that explicit instead.

Fitting scope. `"cooccur"` and `"sbm_random"` read only the Q-matrix and the
concept texts, both of which cover every split, so they are `train_valid_test`.
`"sbm_fit"` counts its observed graph from the training folds only via
`gkt_utils.get_gkt_graph`, so it is `train_folds`. `CGMKT.Inputs.prepare`
reports whichever applies, and `scripts/run_baseline_table.py` will not average
folds across the two.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np


# ---------------------------------------------------------------- observed graphs


def build_kc_cooccurrence(dpath: str, num_c: int) -> np.ndarray:
    """Binary `[num_c, num_c]`: two KCs are linked if a question carries both.

    Static metadata, so this sees every split. On assist2009 the average question
    has 1.20 KCs, which leaves this graph sparse; that sparsity is a property of
    the dataset, not of the construction, and is why `"sbm_fit"` defaults to the
    transition graph instead.
    """
    qmatrix_path = os.path.join(dpath, "qmatrix.npz")
    if not os.path.exists(qmatrix_path):
        raise FileNotFoundError(
            f"CGMKT's KC co-occurrence graph is built from {qmatrix_path}, which "
            f"is missing. Build it with `python scripts/build_qmatrix.py --dataset-name <dataset>`."
        )
    qmatrix = (np.load(qmatrix_path)["matrix"][:, :num_c] > 0).astype(np.float32)
    adjacency = (qmatrix.T @ qmatrix) > 0
    np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.float32)


def build_kc_transition(num_c: int, dpath: str, trainfile: str, *, folds) -> np.ndarray:
    """Above-chance KC-to-KC transitions over the training folds, as GKT counts them.

    Reuses `gkt_utils.get_gkt_graph` so that CGMKT and GKT read the same graph
    from the same counting implementation rather than maintaining a copy.
    Explicit `folds` excludes the held-out validation rows as well as test
    rows, which is what lets `"sbm_fit"` claim `train_folds` scope.

    The thresholding is not cosmetic. `get_gkt_graph` returns a row-stochastic
    matrix, so `> 0` marks every transition that ever occurred; on assist2009
    that is 98% of all KC pairs, and an SBM fitted to it recovers one block of
    density 0.98 and no structure at all (measured: modularity -0.008 over a
    graph of density 0.979). An edge here therefore means "j follows i more
    often than a uniform next-concept would", i.e. `P(j|i) > 1/num_c`, which is
    the weakest threshold that still says something the row normalisation did
    not already fix.
    """
    from .gkt_utils import get_gkt_graph

    if folds is None or not (folds := tuple(folds)):
        raise ValueError("CGMKT sbm_fit requires explicit nonempty training folds")
    graph = get_gkt_graph(
        num_c, dpath, trainfile, testfile=None, graph_type="transition",
        folds=folds,
    )
    graph = np.asarray(graph, dtype=np.float32)
    np.fill_diagonal(graph, 0.0)
    return (graph > (1.0 / max(num_c, 1))).astype(np.float32)


# ---------------------------------------------------------------- SBM, both kinds


def sbm_random_graph(kc_embeddings: np.ndarray, num_clusters: int, seed: int) -> np.ndarray:
    """The released pipeline's graph: one forward pass of an untrained SBMAttention.

    Ported from `cgmkt_preprocess.ipynb` cell 9 (`SBM/sbm_module.py`,
    `generate_sbm_graph`). Kept faithful including the two things that make the
    output near-random -- `softmax` over all k^2 block entries rather than per
    row, and parameters that are never optimised -- because the point of this
    variant is to reproduce upstream, not to improve it.

    The one deliberate change is `seed`: upstream left it unset, so every
    regeneration produced a different graph and no run was reproducible. Fixing
    it costs nothing and is required by AGENTS.md.
    """
    import torch
    import torch.nn as nn

    generator_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        x = torch.from_numpy(np.asarray(kc_embeddings)).float().unsqueeze(0)
        head_dim = x.shape[-1]

        clusters = torch.empty(1, num_clusters, head_dim)
        nn.init.kaiming_normal_(clusters)
        proj = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, head_dim)
        )

        with torch.no_grad():
            # softmax over the flattened k*k block, as upstream: this is what
            # flattens B towards uniform and removes the block structure.
            dist = clusters @ clusters.transpose(-1, -2)
            block = nn.Softmax(dim=-1)(
                dist.reshape(1, num_clusters * num_clusters)
            ).reshape(1, num_clusters, num_clusters)

            q_hat = torch.sigmoid(proj(x) @ clusters.transpose(-1, -2))
            probabilities = q_hat @ (block @ q_hat.transpose(-1, -2))
            sampled = torch.bernoulli(torch.clamp(probabilities + 0.01, 0.0, 1.0))
        return sampled.squeeze(0).squeeze(0).numpy().astype(np.float32)
    finally:
        torch.set_rng_state(generator_state)


def fit_sbm(observed: np.ndarray, num_clusters: int, seed: int, max_iter: int = 100):
    """Fit a stochastic block model to `observed` by Bernoulli likelihood.

    Classification EM, which is the standard cheap SBM fit and is exact enough at
    n=123: spectral initialisation, then alternate
      M-step  `B[a, b]` = edge density between groups a and b,
      E-step  move each node to the group maximising its Bernoulli log-likelihood
              against the current B and the other nodes' assignments,
    until no node moves. Unlike `sbm_random_graph` this is constrained by data:
    the returned B reflects the observed intra/inter densities rather than random
    vector geometry.

    Returns `(membership [num_c, k] one-hot, block [k, k], expected [num_c, num_c])`.
    """
    adjacency = (np.asarray(observed, dtype=np.float32) > 0).astype(np.float32)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    num_nodes = adjacency.shape[0]
    if num_clusters > num_nodes:
        raise ValueError(
            f"CGMKT cannot fit {num_clusters} groups over {num_nodes} KCs."
        )

    labels = _spectral_labels(adjacency, num_clusters, seed)
    for _ in range(max_iter):
        block = _block_densities(adjacency, labels, num_clusters)
        # log B and log(1-B), clipped so an empty block does not give -inf.
        log_p = np.log(np.clip(block, 1e-6, 1 - 1e-6))
        log_q = np.log(np.clip(1.0 - block, 1e-6, 1 - 1e-6))
        one_hot = np.eye(num_clusters, dtype=np.float32)[labels]
        edges = adjacency @ one_hot                       # [n, k] edges to each group
        sizes = one_hot.sum(0)[None, :]                   # [1, k] group sizes
        non_edges = np.maximum(sizes - edges, 0.0)
        # own contribution removed so a node is not scored against itself
        non_edges -= one_hot
        scores = edges @ log_p.T + non_edges @ log_q.T    # [n, k]
        new_labels = scores.argmax(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    block = _block_densities(adjacency, labels, num_clusters)
    membership = np.eye(num_clusters, dtype=np.float32)[labels]
    expected = membership @ block @ membership.T
    np.fill_diagonal(expected, 0.0)
    return membership, block.astype(np.float32), expected.astype(np.float32)


def random_graph(num_c: int, density: float, seed: int) -> np.ndarray:
    """Erdos-Renyi control at a matched density, for the ablation the paper ran."""
    rng = np.random.default_rng(seed)
    adjacency = (rng.random((num_c, num_c)) < density).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


# ---------------------------------------------------------------- groups


def spectral_membership(adjacency: np.ndarray, num_clusters: int, seed: int) -> np.ndarray:
    """One-hot `[num_c, k]` from spectral clustering, as upstream's fallback does.

    `cgmkt.py::_infer_membership_from_adjacency` is what every released run
    actually takes, because no membership file ships. Reproduced here so that
    path is selectable by name rather than reached by accident.
    """
    labels = _spectral_labels(adjacency, num_clusters, seed)
    return np.eye(num_clusters, dtype=np.float32)[labels]


def random_membership(num_c: int, num_clusters: int, seed: int) -> np.ndarray:
    """One-hot `[num_c, k]` with uniformly random assignment, as a control."""
    rng = np.random.default_rng(seed)
    return np.eye(num_clusters, dtype=np.float32)[rng.integers(0, num_clusters, num_c)]


def block_from_membership(adjacency: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Edge density between each pair of groups, for a membership fixed elsewhere.

    Upstream's `_aggregate_block_from_adjacency` equivalent: needed whenever the
    graph and the groups come from different sources.
    """
    labels = np.asarray(membership).argmax(1)
    return _block_densities(
        (np.asarray(adjacency) > 0).astype(np.float32), labels, membership.shape[1]
    )


def group_transition(block: np.ndarray) -> np.ndarray:
    """Row-normalised block matrix with the diagonal removed.

    Eq. (7)'s `T`: self-loops are excluded so the spread mask only ever moves
    mastery to OTHER groups; `g_t` already covers the current one.
    """
    transition = np.asarray(block, dtype=np.float32).copy()
    transition = np.clip(transition, 0.0, None)
    np.fill_diagonal(transition, 0.0)
    row_sum = transition.sum(1, keepdims=True)
    uniform = (1.0 - np.eye(transition.shape[0], dtype=np.float32))
    uniform = uniform / np.maximum(uniform.sum(1, keepdims=True), 1.0)
    return np.where(row_sum > 1e-8, transition / np.maximum(row_sum, 1e-8), uniform)


# ---------------------------------------------------------------- shared internals


def _spectral_labels(adjacency: np.ndarray, num_clusters: int, seed: int) -> np.ndarray:
    """Normalised-Laplacian spectral clustering, matching upstream's fallback.

    Upstream seeds k-means with `torch.linspace` over the node order, which is
    deterministic but arbitrary -- it depends on how the KCs happen to be indexed.
    k-means++ from a seeded RNG is used instead: also reproducible, and not tied
    to the index order.
    """
    adjacency = (np.asarray(adjacency, dtype=np.float32) > 0).astype(np.float32)
    adjacency = (adjacency + adjacency.T) / 2.0
    degree = adjacency.sum(1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    normalised = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    _, vectors = np.linalg.eigh(normalised)
    features = vectors[:, -num_clusters:]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    return _kmeans(features, num_clusters, seed)


def _kmeans(points: np.ndarray, num_clusters: int, seed: int, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = [points[rng.integers(0, points.shape[0])]]
    for _ in range(num_clusters - 1):  # k-means++
        distances = np.min(
            ((points[:, None, :] - np.stack(centres)[None, :, :]) ** 2).sum(-1), axis=1
        )
        total = distances.sum()
        if total <= 0:
            centres.append(points[rng.integers(0, points.shape[0])])
            continue
        centres.append(points[rng.choice(points.shape[0], p=distances / total)])
    centres = np.stack(centres)

    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(iters):
        distances = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new_labels = distances.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(num_clusters):
            if (labels == cluster).any():
                centres[cluster] = points[labels == cluster].mean(0)
    return labels


def _block_densities(adjacency: np.ndarray, labels: np.ndarray, num_clusters: int):
    """`[k, k]` edge density between groups, with the diagonal counted over pairs."""
    one_hot = np.eye(num_clusters, dtype=np.float32)[labels]
    edges = one_hot.T @ adjacency @ one_hot
    sizes = one_hot.sum(0)
    pairs = np.outer(sizes, sizes) - np.diag(sizes)  # exclude self-pairs on the diagonal
    return edges / np.maximum(pairs, 1.0)


def row_normalise(adjacency: np.ndarray) -> np.ndarray:
    """Row-stochastic, as upstream's `row_normalize` (zero rows left at zero)."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    row_sum = adjacency.sum(1, keepdims=True)
    return adjacency / np.maximum(row_sum, 1e-8)


def incidence_question_graph(dpath: str, num_q: int, num_c: int):
    """The released `ques_skill_gcn_adj.pt`: Q-matrix padded to `[num_q, num_q]`.

    Rebuilt from `qmatrix.npz` rather than taken from the authors' Google Drive
    file, for the reason `kc_graph_utils` gives: a downloaded tensor cannot be
    checked against the dataset this repo actually preprocessed. Verified to
    match the released file's structure on assist2009 -- square, non-zero columns
    confined to `0..num_c-1`, row-normalised, asymmetric.
    """
    import scipy.sparse as sp
    import torch

    qmatrix_path = os.path.join(dpath, "qmatrix.npz")
    qmatrix = (np.load(qmatrix_path)["matrix"][:num_q, :num_c] > 0).astype(np.float32)
    padded = sp.csr_matrix((num_q, num_q), dtype=np.float32)
    padded[:, :num_c] = sp.csr_matrix(qmatrix)
    row_sum = np.asarray(padded.sum(1)).ravel()
    scale = sp.diags((1.0 / np.maximum(row_sum, 1.0)).astype(np.float32))
    normalised = (scale @ padded).tocoo()
    indices = torch.tensor(np.vstack([normalised.row, normalised.col]), dtype=torch.long)
    values = torch.tensor(normalised.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(num_q, num_q)).coalesce()


def fingerprint(*parts) -> str:
    """Short digest for cache names, as gkt.py and kc_graph_utils.py use."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]
