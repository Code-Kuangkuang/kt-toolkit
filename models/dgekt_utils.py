from itertools import zip_longest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def build_dgekt_graphs(
    dpath, sequence_file, num_q, num_c, train_folds, association_files=None
):
    """Build DGEKT's dual graphs from toolkit sequence data.

    Question-concept associations are static metadata, so the hypergraph uses
    every row in the train/validation sequence file. Response-conditioned
    transitions are learned only from the current fold's training rows.
    """
    if int(num_q) <= 0 or int(num_c) <= 0:
        raise ValueError(f"DGEKT requires positive num_q and num_c, got {num_q}, {num_c}.")

    sequence_path = _resolve_path(dpath, sequence_file)

    columns = ["questions", "concepts", "responses"]
    header = pd.read_csv(sequence_path, nrows=0).columns
    if "fold" in header:
        columns.append("fold")
    frame = pd.read_csv(sequence_path, usecols=columns, dtype=str, keep_default_na=False)
    if "fold" not in frame.columns:
        raise KeyError(f"DGEKT graph source {sequence_path} must contain a 'fold' column.")

    train_fold_set = {int(fold) for fold in train_folds}
    frame["fold"] = frame["fold"].astype(int)
    transition_frame = frame[frame["fold"].isin(train_fold_set)]
    if transition_frame.empty:
        raise ValueError(
            f"No DGEKT graph rows found in {sequence_path} for folds {sorted(train_fold_set)}."
        )

    association_paths = [sequence_path]
    association_frames = [frame[["questions", "concepts"]]]
    for association_file in association_files or []:
        association_path = _resolve_path(dpath, association_file)
        if association_path in association_paths:
            continue
        association_paths.append(association_path)
        association_frames.append(
            pd.read_csv(
                association_path,
                usecols=["questions", "concepts"],
                dtype=str,
                keep_default_na=False,
            )
        )

    associations = set()
    for association_frame in association_frames:
        for questions, concepts in zip(
            association_frame["questions"], association_frame["concepts"]
        ):
            for question, concept_token in zip_longest(
                _parse_ids(questions), str(concepts).split(","), fillvalue=None
            ):
                if question is None or not 0 <= question < num_q:
                    continue
                for concept in _parse_concepts(concept_token):
                    if 0 <= concept < num_c:
                        associations.add((question, concept))

    if not associations:
        raise ValueError(f"No valid question-concept associations found in {sequence_path}.")

    hypergraph = _build_hypergraph(num_q, num_c, associations)
    transition_out, transition_in, transition_count = _build_transition_graphs(
        transition_frame, num_q
    )

    covered_questions = len({question for question, _ in associations})
    stats = {
        "source_file": sequence_path,
        "association_files": association_paths,
        "train_folds": sorted(train_fold_set),
        "num_nodes": 2 * int(num_q),
        "num_hyperedges": 2 * int(num_c),
        "question_concept_pairs": len(associations),
        "covered_questions": covered_questions,
        "uncovered_questions": int(num_q) - covered_questions,
        "transition_count": int(transition_count),
        "transition_edges": int(transition_out._nnz() - 2 * int(num_q)),
    }
    return hypergraph, transition_out, transition_in, stats


def _resolve_path(dpath, filename):
    from pathlib import Path

    path = Path(str(filename))
    if not path.is_absolute() and dpath:
        path = Path(dpath) / path
    return str(path)


def _parse_ids(value):
    result = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(float(token)))
        except ValueError:
            result.append(-1)
    return result


def _parse_concepts(value):
    result = []
    for token in str(value).split("_"):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(float(token)))
        except ValueError:
            continue
    return result


def _build_hypergraph(num_q, num_c, associations):
    rows = []
    cols = []
    for question, concept in associations:
        rows.extend((question, question + num_q))
        cols.extend((concept, concept + num_c))

    values = np.ones(len(rows), dtype=np.float32)
    incidence = sp.coo_matrix(
        (values, (rows, cols)), shape=(2 * num_q, 2 * num_c), dtype=np.float32
    ).tocsr()

    node_degree = np.asarray(incidence.sum(axis=1)).reshape(-1)
    edge_degree = np.asarray(incidence.sum(axis=0)).reshape(-1)
    node_scale = np.zeros_like(node_degree, dtype=np.float32)
    edge_scale = np.zeros_like(edge_degree, dtype=np.float32)
    node_scale[node_degree > 0] = np.power(node_degree[node_degree > 0], -0.5)
    edge_scale[edge_degree > 0] = np.power(edge_degree[edge_degree > 0], -0.5)

    # Keep the normalized incidence matrix N = Dv^-1/2 H De^-1/2 instead of
    # materializing G = N N^T. The latter can contain billions of entries when
    # many questions share a concept. Hypergraph convolution computes
    # G X associatively as N (N^T X), which is mathematically equivalent.
    normalized = (sp.diags(node_scale) @ incidence @ sp.diags(edge_scale)).tocsr()
    return _scipy_to_torch_sparse(normalized)


def _build_transition_graphs(frame, num_q):
    rows = []
    cols = []
    transition_count = 0
    for _, row in frame.iterrows():
        interactions = []
        for question, response in zip(_parse_ids(row["questions"]), _parse_ids(row["responses"])):
            if 0 <= question < num_q and response in (0, 1):
                # Match the reference implementation: correct nodes occupy
                # [0, num_q), incorrect nodes occupy [num_q, 2 * num_q).
                interactions.append(question + num_q * (1 - response))
        for current, nxt in zip(interactions, interactions[1:]):
            rows.append(current)
            cols.append(nxt)
            transition_count += 1

    node_count = 2 * num_q
    values = np.ones(len(rows), dtype=np.float32)
    adjacency = sp.coo_matrix(
        (values, (rows, cols)), shape=(node_count, node_count), dtype=np.float32
    ).tocsr()
    adjacency.sum_duplicates()
    reverse = adjacency.T.tocsr()
    identity = sp.eye(node_count, dtype=np.float32, format="csr")
    return (
        _scipy_to_torch_sparse(_row_normalize(adjacency + identity)),
        _scipy_to_torch_sparse(_row_normalize(reverse + identity)),
        transition_count,
    )


def _row_normalize(matrix):
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    inverse[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    return (sp.diags(inverse) @ matrix).tocsr()


def _scipy_to_torch_sparse(matrix):
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
