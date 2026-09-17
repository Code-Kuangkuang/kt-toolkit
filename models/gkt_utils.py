import os

import numpy as np
import pandas as pd
import torch


def get_gkt_graph(num_c, dpath, trainfile, testfile=None, graph_type="dense",
                  tofile="graph.npz", folds=None):
    """Build and cache the graph used by GKT.

    `folds=None` matches pyKT's init flow, which counts transitions from the
    training and test files together. Passing the current fold's training folds
    restricts it to those rows and drops the test file, which is what AGENTS.md
    requires of a derived feature: the graph is part of the model's structure,
    and building it from the test split makes the run transductive.

    Only the `concepts` column is read either way, so neither is label leakage.
    """
    if graph_type == "dense":
        graph = build_dense_graph(num_c)
    elif graph_type == "transition":
        frames = []
        fold_set = {int(f) for f in folds} if folds is not None else None
        sources = (trainfile,) if fold_set is not None else (trainfile, testfile)
        for filename in sources:
            if not filename:
                continue
            path = os.path.join(dpath, filename)
            if not os.path.exists(path):
                continue
            frame = pd.read_csv(path)
            if fold_set is not None:
                if "fold" not in frame.columns:
                    raise ValueError(
                        f"GKT graph source {path} has no 'fold' column, so it "
                        "cannot be restricted to the training folds. Set "
                        "`pykt_transductive: true` to build from every split."
                    )
                frame = frame[frame["fold"].astype(int).isin(fold_set)]
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise FileNotFoundError(
                f"Cannot build GKT transition graph: no source rows found in {dpath}"
                + (f" for folds {sorted(fold_set)}." if fold_set is not None else ".")
            )
        graph = build_transition_graph(pd.concat(frames, ignore_index=True), num_c)
    else:
        raise ValueError(f"Unsupported GKT graph_type: {graph_type}")

    os.makedirs(dpath, exist_ok=True)
    matrix = graph.detach().cpu().numpy() if torch.is_tensor(graph) else graph
    np.savez(os.path.join(dpath, tofile), matrix=matrix)
    return graph


def build_transition_graph(df, concept_num):
    graph = np.zeros((concept_num, concept_num), dtype=np.float32)
    for _, row in df.iterrows():
        concepts = list(_iter_first_concepts(row.get("concepts", "")))
        for pre, nxt in zip(concepts, concepts[1:]):
            if 0 <= pre < concept_num and 0 <= nxt < concept_num:
                graph[pre, nxt] += 1

    np.fill_diagonal(graph, 0)
    rowsum = graph.sum(axis=1)
    nonzero = rowsum != 0
    graph[nonzero] = graph[nonzero] / rowsum[nonzero, None]
    return torch.from_numpy(graph).float()


def build_dense_graph(concept_num):
    if concept_num <= 1:
        return torch.zeros((concept_num, concept_num), dtype=torch.float32)
    graph = np.full((concept_num, concept_num), 1.0 / (concept_num - 1), dtype=np.float32)
    np.fill_diagonal(graph, 0)
    return torch.from_numpy(graph).float()


def _iter_first_concepts(value):
    for token in str(value).split(","):
        token = token.strip()
        if not token or token == "-1" or token.lower() == "nan":
            continue
        first = token.split("_", 1)[0]
        if first == "-1":
            continue
        yield int(first)
