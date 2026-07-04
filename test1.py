import torch
import torch.nn as nn

import dhg

from dhg.structure.hypergraphs import Hypergraph


def build_feature_knn_hypergraph(X: torch.Tensor, k: int, device=None) -> Hypergraph:
    """Build a dhg KNN hypergraph from feature representations."""
    if X.dim() != 2:
        raise ValueError("X must be a 2D tensor of shape [num_samples, num_features]")

    num_samples = X.shape[0]
    if num_samples <= 0:
        raise ValueError("X must contain at least one sample")

    device = torch.device(device or X.device)
    with torch.no_grad():
        X_cpu = X.detach().cpu()
        self_loop_edges = [[i] for i in range(num_samples)]
        if int(k) <= 0 or num_samples == 1:
            hg = dhg.Hypergraph(num_v=num_samples, e_list=self_loop_edges)
        else:
            safe_k = min(int(k), max(num_samples - 1, 1))
            hg = dhg.Hypergraph.from_feature_kNN(X_cpu, k=safe_k)
            hg.add_hyperedges(self_loop_edges, group_name="self_loop")
        hg = hg.to(device)
    return hg

