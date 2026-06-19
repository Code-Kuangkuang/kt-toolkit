import os

import numpy as np
import pandas as pd
import torch


def get_gkt_graph(num_c, dpath, trainfile, testfile=None, graph_type="dense", tofile="graph.npz"):
    """Build and cache the graph used by GKT, matching pykt's init flow."""
    if graph_type == "dense":
        graph = build_dense_graph(num_c)
    elif graph_type == "transition":
        frames = []
        for filename in (trainfile, testfile):
            if not filename:
                continue
            path = os.path.join(dpath, filename)
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
        if not frames:
            raise FileNotFoundError(
                f"Cannot build GKT transition graph: no source CSV found in {dpath}."
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
