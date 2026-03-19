"""Node2Vec-based phenotype matching via graph embeddings.

Trains node2vec embeddings on the heterogeneous knowledge graph (HPO DAG +
disease + gene nodes). Ranking: given query HPO terms, compute average
embedding of query terms, rank diseases by cosine similarity.

Implementation uses biased random walks (Grover & Leskovec 2016) and
a lightweight skip-gram model to avoid heavy dependencies (no gensim).

Memory-safe: walks are generated in batches, embeddings stored as dense
numpy arrays.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.node2vec_matcher
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from bioagentics.diagnostics.rare_disease.ic_matcher import RankResult

logger = logging.getLogger(__name__)


@dataclass
class Node2VecConfig:
    """Configuration for node2vec training."""

    dimensions: int = 64
    walk_length: int = 30
    num_walks: int = 10
    p: float = 1.0  # Return parameter
    q: float = 1.0  # In-out parameter
    window_size: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 0.001
    epochs: int = 5
    negative_samples: int = 5
    seed: int = 42


def _build_adjacency(graph: nx.DiGraph) -> dict[str, list[str]]:
    """Build adjacency list treating graph as undirected."""
    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in graph.edges():
        adj[u].append(v)
        adj[v].append(u)
    # Deduplicate
    return {node: list(set(neighbors)) for node, neighbors in adj.items()}


def _biased_walk(
    adj: dict[str, list[str]],
    start: str,
    walk_length: int,
    p: float,
    q: float,
    rng: random.Random,
) -> list[str]:
    """Generate a single biased random walk from a start node.

    p controls likelihood of returning to previous node.
    q controls exploration vs exploitation.
    """
    walk = [start]

    if start not in adj or not adj[start]:
        return walk

    # First step: uniform random
    walk.append(rng.choice(adj[start]))

    for _ in range(walk_length - 2):
        cur = walk[-1]
        prev = walk[-2]

        neighbors = adj.get(cur, [])
        if not neighbors:
            break

        prev_neighbors = set(adj.get(prev, []))

        # Compute unnormalized transition probabilities
        weights = []
        for nbr in neighbors:
            if nbr == prev:
                weights.append(1.0 / p)  # Return to previous
            elif nbr in prev_neighbors:
                weights.append(1.0)  # Neighbor of previous (BFS-like)
            else:
                weights.append(1.0 / q)  # Move away (DFS-like)

        # Normalize and sample
        total = sum(weights)
        r = rng.random() * total
        cumsum = 0.0
        chosen = neighbors[0]
        for nbr, w in zip(neighbors, weights):
            cumsum += w
            if r <= cumsum:
                chosen = nbr
                break

        walk.append(chosen)

    return walk


def generate_walks(
    graph: nx.DiGraph,
    config: Node2VecConfig,
) -> list[list[str]]:
    """Generate biased random walks on the graph.

    Returns:
        List of walks, each walk is a list of node IDs.
    """
    adj = _build_adjacency(graph)
    rng = random.Random(config.seed)
    nodes = list(adj.keys())

    walks: list[list[str]] = []
    for _ in range(config.num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = _biased_walk(
                adj, node, config.walk_length, config.p, config.q, rng
            )
            if len(walk) > 1:
                walks.append(walk)

    logger.info(
        "Generated %d walks (avg length %.1f)",
        len(walks),
        np.mean([len(w) for w in walks]) if walks else 0,
    )
    return walks


def _build_vocab(walks: list[list[str]]) -> tuple[dict[str, int], list[str]]:
    """Build vocabulary from walks. Returns (word2idx, idx2word)."""
    vocab: set[str] = set()
    for walk in walks:
        vocab.update(walk)
    idx2word = sorted(vocab)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def _compute_noise_distribution(walks: list[list[str]], word2idx: dict[str, int]) -> np.ndarray:
    """Compute unigram noise distribution raised to 3/4 power (Mikolov et al.)."""
    counts = np.zeros(len(word2idx), dtype=np.float64)
    for walk in walks:
        for token in walk:
            counts[word2idx[token]] += 1
    counts = np.power(counts, 0.75)
    return counts / counts.sum()


def train_embeddings(
    walks: list[list[str]],
    config: Node2VecConfig,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Train skip-gram embeddings from random walks.

    Uses negative sampling with SGD. Lightweight implementation
    suitable for moderate-sized graphs.

    Returns:
        Tuple of (embeddings array [vocab_size, dimensions], word2idx, idx2word).
    """
    word2idx, idx2word = _build_vocab(walks)
    vocab_size = len(word2idx)

    if vocab_size == 0:
        return np.zeros((0, config.dimensions)), word2idx, idx2word

    rng = np.random.RandomState(config.seed)

    # Initialize embeddings
    W = rng.uniform(-0.5 / config.dimensions, 0.5 / config.dimensions,
                     (vocab_size, config.dimensions)).astype(np.float32)
    C = np.zeros_like(W)  # Context embeddings

    noise_dist = _compute_noise_distribution(walks, word2idx)

    # Collect all skip-gram pairs
    pairs: list[tuple[int, int]] = []
    for walk in walks:
        indices = [word2idx[w] for w in walk]
        for i, center_idx in enumerate(indices):
            window_start = max(0, i - config.window_size)
            window_end = min(len(indices), i + config.window_size + 1)
            for j in range(window_start, window_end):
                if j != i:
                    pairs.append((center_idx, indices[j]))

    if not pairs:
        return W, word2idx, idx2word

    logger.info(
        "Training skip-gram: vocab=%d, pairs=%d, dims=%d, epochs=%d",
        vocab_size, len(pairs), config.dimensions, config.epochs,
    )

    pair_array = np.array(pairs, dtype=np.int32)
    n_pairs = len(pair_array)

    for epoch in range(config.epochs):
        # Shuffle pairs
        perm = rng.permutation(n_pairs)
        lr = config.learning_rate - (
            (config.learning_rate - config.min_learning_rate)
            * epoch / max(config.epochs - 1, 1)
        )

        total_loss = 0.0

        for idx in perm:
            center, context = pair_array[idx]

            # Positive sample
            dot = np.dot(W[center], C[context])
            sig = 1.0 / (1.0 + np.exp(-np.clip(dot, -6, 6)))
            grad = lr * (1.0 - sig)
            total_loss += -np.log(max(sig, 1e-10))

            # Update
            W[center] += grad * C[context]
            C[context] += grad * W[center]

            # Negative samples
            neg_indices = rng.choice(
                vocab_size, size=config.negative_samples, p=noise_dist
            )
            for neg_idx in neg_indices:
                if neg_idx == context:
                    continue
                dot = np.dot(W[center], C[neg_idx])
                sig = 1.0 / (1.0 + np.exp(-np.clip(dot, -6, 6)))
                grad = lr * (-sig)
                total_loss += -np.log(max(1.0 - sig + 1e-10, 1e-10))

                W[center] += grad * C[neg_idx]
                C[neg_idx] += grad * W[center]

        avg_loss = total_loss / n_pairs
        logger.info("Epoch %d/%d: avg_loss=%.4f, lr=%.5f", epoch + 1, config.epochs, avg_loss, lr)

    return W, word2idx, idx2word


@dataclass
class Node2VecMatcher:
    """Node2Vec-based disease ranking using graph embeddings.

    Attributes:
        embeddings: Embedding matrix [vocab_size, dimensions].
        word2idx: Mapping from node ID to embedding index.
        idx2word: Reverse mapping from index to node ID.
    """

    embeddings: np.ndarray
    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: list[str] = field(default_factory=list)

    def get_embedding(self, node_id: str) -> np.ndarray | None:
        """Get the embedding vector for a node."""
        idx = self.word2idx.get(node_id)
        if idx is None:
            return None
        return self.embeddings[idx]

    def query_embedding(self, hpo_terms: list[str]) -> np.ndarray | None:
        """Compute average embedding for a set of query HPO terms."""
        vectors = []
        for term in hpo_terms:
            vec = self.get_embedding(term)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    @classmethod
    def from_graph(
        cls,
        graph: nx.DiGraph,
        config: Node2VecConfig | None = None,
    ) -> Node2VecMatcher:
        """Train node2vec embeddings on a graph and return a matcher."""
        if config is None:
            config = Node2VecConfig()

        walks = generate_walks(graph, config)
        embeddings, word2idx, idx2word = train_embeddings(walks, config)

        return cls(embeddings=embeddings, word2idx=word2idx, idx2word=idx2word)


def rank_diseases(
    matcher: Node2VecMatcher,
    query_hpo_terms: list[str],
    disease_ids: list[str],
) -> list[RankResult]:
    """Rank diseases by cosine similarity to query embedding.

    Args:
        matcher: Node2VecMatcher with trained embeddings.
        query_hpo_terms: Patient's HPO term IDs.
        disease_ids: List of disease IDs to rank.

    Returns:
        List of RankResult sorted by score descending.
    """
    query_vec = matcher.query_embedding(query_hpo_terms)
    if query_vec is None:
        return [
            RankResult(disease_id=d, score=0.0, rank=i + 1)
            for i, d in enumerate(disease_ids)
        ]

    query_vec_2d = query_vec.reshape(1, -1)

    results: list[RankResult] = []
    for disease_id in disease_ids:
        disease_vec = matcher.get_embedding(disease_id)
        if disease_vec is None:
            results.append(RankResult(disease_id=disease_id, score=0.0))
            continue

        sim = cosine_similarity(query_vec_2d, disease_vec.reshape(1, -1))[0, 0]
        results.append(RankResult(disease_id=disease_id, score=float(sim)))

    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
