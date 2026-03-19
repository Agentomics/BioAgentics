"""Graph Attention Network (GAT) for disease-phenotype link prediction.

Trains a GAT on the heterogeneous knowledge graph for disease-phenotype
link prediction. Score a query phenotype set against all diseases by
aggregating link prediction probabilities.

Implements GAT layers in pure PyTorch (no PyTorch Geometric dependency)
for portability and memory efficiency.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.gat_matcher
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bioagentics.diagnostics.rare_disease.ic_matcher import RankResult

logger = logging.getLogger(__name__)


@dataclass
class GATConfig:
    """Configuration for the GAT model."""

    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.005
    epochs: int = 50
    negative_ratio: int = 3  # negative edges per positive edge
    seed: int = 42


class GATLayer(nn.Module):
    """Single Graph Attention layer (Velickovic et al. 2018)."""

    def __init__(self, in_features: int, out_features: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads

        self.W = nn.Linear(in_features, num_heads * self.out_per_head, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.out_per_head))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.out_per_head))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_features].
            edge_index: Edge indices [2, num_edges] (src, dst).

        Returns:
            Updated node features [num_nodes, num_heads * out_per_head].
        """
        num_nodes = x.size(0)

        # Linear transform: [N, H * out_per_head]
        h = self.W(x).view(num_nodes, self.num_heads, self.out_per_head)

        # Attention scores
        src, dst = edge_index[0], edge_index[1]
        # [E, H]
        attn_src = (h[src] * self.a_src.unsqueeze(0)).sum(dim=-1)
        attn_dst = (h[dst] * self.a_dst.unsqueeze(0)).sum(dim=-1)
        attn = self.leaky_relu(attn_src + attn_dst)

        # Sparse softmax per destination node
        attn_max = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attn_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(attn), attn, reduce="amax")
        attn = torch.exp(attn - attn_max[dst])
        attn_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(attn), attn)
        attn = attn / (attn_sum[dst] + 1e-10)
        attn = self.dropout(attn)

        # Weighted message passing
        msg = h[src] * attn.unsqueeze(-1)  # [E, H, out_per_head]
        out = torch.zeros(num_nodes, self.num_heads, self.out_per_head, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        return out.reshape(num_nodes, -1)


class GATLinkPredictor(nn.Module):
    """GAT-based link prediction model for disease-phenotype scoring."""

    def __init__(self, num_nodes: int, config: GATConfig):
        super().__init__()
        self.config = config

        # Learnable node embeddings (input features)
        self.node_embed = nn.Embedding(num_nodes, config.hidden_dim)

        # GAT layers
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                GATLayer(config.hidden_dim, config.hidden_dim, config.num_heads, config.dropout)
            )

        # Link prediction head: dot product + MLP
        self.link_pred = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.node_embed.weight)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node representations via GAT layers."""
        x = self.node_embed.weight
        for layer in self.layers:
            x = F.elu(layer(x, edge_index)) + x  # Residual
        return x

    def predict_link(
        self,
        z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probability for given source-destination pairs.

        Returns logits (not probabilities).
        """
        h = torch.cat([z[src], z[dst]], dim=-1)
        return self.link_pred(h).squeeze(-1)


def _prepare_graph_data(
    graph: nx.DiGraph,
) -> tuple[dict[str, int], torch.Tensor, list[tuple[int, int]]]:
    """Convert NetworkX graph to tensors for GAT.

    Returns:
        Tuple of (node2idx, edge_index tensor, has_phenotype edge list).
    """
    nodes = list(graph.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}

    # Build undirected edge index (GAT operates on undirected)
    edges_set: set[tuple[int, int]] = set()
    has_phenotype_edges: list[tuple[int, int]] = []

    for u, v, data in graph.edges(data=True):
        ui, vi = node2idx[u], node2idx[v]
        edges_set.add((ui, vi))
        edges_set.add((vi, ui))

        if data.get("edge_type") == "has_phenotype":
            has_phenotype_edges.append((ui, vi))

    edge_list = list(edges_set)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return node2idx, edge_index, has_phenotype_edges


def _sample_negative_edges(
    num_nodes: int,
    positive_edges: list[tuple[int, int]],
    num_negatives: int,
    rng: np.random.RandomState,
) -> list[tuple[int, int]]:
    """Sample negative (non-existing) edges."""
    pos_set = set(positive_edges)
    negatives: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = num_negatives * 10

    while len(negatives) < num_negatives and attempts < max_attempts:
        src = rng.randint(0, num_nodes)
        dst = rng.randint(0, num_nodes)
        if src != dst and (src, dst) not in pos_set:
            negatives.append((src, dst))
            pos_set.add((src, dst))  # Prevent duplicates
        attempts += 1

    return negatives


def train_gat(
    graph: nx.DiGraph,
    config: GATConfig | None = None,
) -> tuple[GATLinkPredictor, dict[str, int], list[float]]:
    """Train a GAT model for link prediction on the knowledge graph.

    Args:
        graph: Heterogeneous knowledge graph.
        config: Training configuration.

    Returns:
        Tuple of (trained model, node2idx mapping, loss history).
    """
    if config is None:
        config = GATConfig()

    torch.manual_seed(config.seed)
    rng = np.random.RandomState(config.seed)

    node2idx, edge_index, pos_edges = _prepare_graph_data(graph)
    num_nodes = len(node2idx)

    if not pos_edges:
        logger.warning("No has_phenotype edges found for training")
        model = GATLinkPredictor(num_nodes, config)
        return model, node2idx, []

    model = GATLinkPredictor(num_nodes, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_history: list[float] = []

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        # Encode
        z = model.encode(edge_index)

        # Positive samples
        pos_src = torch.tensor([e[0] for e in pos_edges], dtype=torch.long)
        pos_dst = torch.tensor([e[1] for e in pos_edges], dtype=torch.long)
        pos_logits = model.predict_link(z, pos_src, pos_dst)

        # Negative samples
        neg_edges = _sample_negative_edges(
            num_nodes, pos_edges,
            len(pos_edges) * config.negative_ratio, rng,
        )
        neg_src = torch.tensor([e[0] for e in neg_edges], dtype=torch.long)
        neg_dst = torch.tensor([e[1] for e in neg_edges], dtype=torch.long)
        neg_logits = model.predict_link(z, neg_src, neg_dst)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, config.epochs, loss_val)

    return model, node2idx, loss_history


@dataclass
class GATMatcher:
    """GAT-based disease ranking using link prediction scores.

    Attributes:
        model: Trained GATLinkPredictor.
        node2idx: Mapping from node ID to index.
        idx2node: Reverse mapping.
        graph_edge_index: Edge index tensor for encoding.
    """

    model: GATLinkPredictor
    node2idx: dict[str, int] = field(default_factory=dict)
    idx2node: dict[int, str] = field(default_factory=dict)
    graph_edge_index: torch.Tensor = field(default_factory=lambda: torch.empty(2, 0, dtype=torch.long))

    @classmethod
    def from_graph(
        cls,
        graph: nx.DiGraph,
        config: GATConfig | None = None,
    ) -> GATMatcher:
        """Train a GAT model on a graph and return a matcher."""
        model, node2idx, _ = train_gat(graph, config)
        idx2node = {i: n for n, i in node2idx.items()}
        _, edge_index, _ = _prepare_graph_data(graph)

        return cls(
            model=model,
            node2idx=node2idx,
            idx2node=idx2node,
            graph_edge_index=edge_index,
        )

    @torch.no_grad()
    def score_disease_phenotype(
        self,
        disease_id: str,
        phenotype_id: str,
    ) -> float:
        """Get link prediction score for a disease-phenotype pair."""
        if disease_id not in self.node2idx or phenotype_id not in self.node2idx:
            return 0.0

        self.model.eval()
        z = self.model.encode(self.graph_edge_index)

        src = torch.tensor([self.node2idx[disease_id]], dtype=torch.long)
        dst = torch.tensor([self.node2idx[phenotype_id]], dtype=torch.long)
        logit = self.model.predict_link(z, src, dst)
        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def score_disease(
        self,
        query_hpo_terms: list[str],
        disease_id: str,
    ) -> float:
        """Score a disease against a query phenotype profile.

        Aggregates link prediction probabilities across query terms.
        """
        if disease_id not in self.node2idx:
            return 0.0

        valid_terms = [t for t in query_hpo_terms if t in self.node2idx]
        if not valid_terms:
            return 0.0

        self.model.eval()
        z = self.model.encode(self.graph_edge_index)

        disease_idx = self.node2idx[disease_id]
        src = torch.tensor([disease_idx] * len(valid_terms), dtype=torch.long)
        dst = torch.tensor([self.node2idx[t] for t in valid_terms], dtype=torch.long)
        logits = self.model.predict_link(z, src, dst)
        probs = torch.sigmoid(logits)

        return probs.mean().item()


def rank_diseases(
    matcher: GATMatcher,
    query_hpo_terms: list[str],
    disease_ids: list[str],
) -> list[RankResult]:
    """Rank diseases by GAT link prediction scores.

    Args:
        matcher: GATMatcher with trained model.
        query_hpo_terms: Patient's HPO term IDs.
        disease_ids: List of disease IDs to rank.

    Returns:
        List of RankResult sorted by score descending.
    """
    results: list[RankResult] = []

    for disease_id in disease_ids:
        score = matcher.score_disease(query_hpo_terms, disease_id)
        results.append(RankResult(disease_id=disease_id, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
