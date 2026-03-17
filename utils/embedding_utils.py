"""Word2Vec embedding loading, similarity search, and OOV handling utilities.

This module provides helpers for working with the Word2Vec embedding space
used as the training target and inference class-proxy for sign language
recognition.

Key utilities:

- :func:`load_embeddings` — load the precomputed ``.npy`` embedding matrix.
- :func:`load_vocab` — load the ordered class name list from ``vocab.json``.
- :func:`nearest_neighbour` — retrieve top-k class labels for a predicted
  embedding via cosine similarity.
- :func:`embedding_coverage` — audit which vocabulary words are OOV in a
  Word2Vec model.
- :func:`mean_pairwise_similarity` — measure how well-separated the class
  embeddings are (useful for diagnosing confusion-prone word pairs).

Example::

    from utils.embedding_utils import load_embeddings, load_vocab, nearest_neighbour
    import torch

    vocab = load_vocab("dataset/annotations/vocab.json")
    emb_matrix = load_embeddings("dataset/annotations/word2vec_embeddings.npy")

    pred = torch.randn(1, 300)
    labels, scores = nearest_neighbour(pred, emb_matrix, vocab, k=5)
    # labels: [["book", "read", ...]], scores: tensor([[0.91, 0.87, ...]])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


# =============================================================================
# Loading helpers
# =============================================================================


def load_vocab(vocab_path: str | Path) -> list[str]:
    """Load the ordered class vocabulary from ``vocab.json``.

    The vocab file may be either a plain JSON array (the format written by
    ``build_annotations.py``) or the stub dict format used during development.

    Args:
        vocab_path: Path to ``vocab.json``.

    Returns:
        Ordered list of class name strings.  The index of a name in this
        list equals the ``label_idx`` used by the model and Dataset.

    Raises:
        FileNotFoundError: If ``vocab_path`` does not exist.
        ValueError: If the file does not contain a recognisable format.
    """
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        if "words" in raw:
            return raw["words"]
        raise ValueError(
            f"vocab.json dict does not contain a 'words' key. " f"Keys found: {list(raw.keys())}"
        )

    raise ValueError(
        f"vocab.json must be a JSON array or dict with 'words' key, " f"got {type(raw).__name__}."
    )


def load_embeddings(
    embeddings_path: str | Path,
    as_tensor: bool = True,
    device: str | None = None,
) -> Tensor | np.ndarray:
    """Load the precomputed Word2Vec embedding matrix.

    Args:
        embeddings_path: Path to ``word2vec_embeddings.npy`` — a float32
            NumPy array of shape ``(num_classes, embedding_dim)``.
        as_tensor: If ``True``, return a ``torch.Tensor``; otherwise return
            the raw NumPy array.
        device: Device to place the tensor on (e.g. ``"cuda"``).  Ignored
            when ``as_tensor=False``.

    Returns:
        Float32 embedding matrix of shape ``(num_classes, embedding_dim)``,
        L2-normalised row-wise, as a tensor or NumPy array.

    Raises:
        FileNotFoundError: If the ``.npy`` file does not exist.
    """
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding matrix not found: {embeddings_path}\n"
            "Run dataset/annotations/build_annotations.py first."
        )

    arr = np.load(str(embeddings_path)).astype(np.float32)
    log.debug("Loaded embeddings — shape: %s  dtype: %s", arr.shape, arr.dtype)

    if not as_tensor:
        return arr

    tensor = torch.from_numpy(arr)
    if device:
        tensor = tensor.to(device)
    return tensor


def load_embeddings_and_vocab(
    embeddings_path: str | Path,
    vocab_path: str | Path,
    device: str | None = None,
) -> tuple[Tensor, list[str]]:
    """Convenience loader that returns both the embedding matrix and vocab.

    Args:
        embeddings_path: Path to ``word2vec_embeddings.npy``.
        vocab_path: Path to ``vocab.json``.
        device: Device for the returned tensor.

    Returns:
        A tuple of:
            - ``embeddings``: Float32 tensor ``(num_classes, embedding_dim)``.
            - ``vocab``: Ordered list of class name strings.
    """
    vocab = load_vocab(vocab_path)
    embeddings = load_embeddings(embeddings_path, as_tensor=True, device=device)
    return embeddings, vocab


# =============================================================================
# Nearest-neighbour retrieval
# =============================================================================


def nearest_neighbour(
    pred_embeddings: Tensor,
    class_embeddings: Tensor,
    vocab: list[str],
    k: int = 5,
) -> tuple[list[list[str]], Tensor]:
    """Retrieve top-k class labels for predicted embeddings via cosine similarity.

    Both ``pred_embeddings`` and ``class_embeddings`` are assumed to be
    L2-normalised, so cosine similarity reduces to a dot product.

    Args:
        pred_embeddings: Predicted embedding batch ``(B, D)``, L2-normalised.
        class_embeddings: Class embedding matrix ``(C, D)``, L2-normalised.
        vocab: Ordered class name list of length ``C``.
        k: Number of top predictions to return per sample.

    Returns:
        A tuple of:
            - ``labels``: List of ``B`` lists, each containing ``k`` class
              name strings ordered by decreasing similarity.
            - ``scores``: Float32 tensor of shape ``(B, k)`` with cosine
              similarity scores.

    Raises:
        ValueError: If ``k`` exceeds the number of classes.

    Example::

        labels, scores = nearest_neighbour(pred, class_matrix, vocab, k=5)
        print(labels[0])   # ["book", "read", "library", "study", "paper"]
    """
    num_classes = class_embeddings.shape[0]
    if k > num_classes:
        raise ValueError(f"k={k} exceeds the number of classes ({num_classes}).")

    # (B, D) × (D, C) → (B, C) cosine similarities (both sides unit-norm)
    similarities = pred_embeddings @ class_embeddings.T  # (B, C)
    top_scores, top_indices = similarities.topk(k=k, dim=-1, largest=True)

    labels: list[list[str]] = [[vocab[idx.item()] for idx in row] for row in top_indices]
    return labels, top_scores


def top1_accuracy(
    pred_embeddings: Tensor,
    class_embeddings: Tensor,
    true_label_indices: Tensor,
) -> float:
    """Compute top-1 nearest-neighbour accuracy over a batch.

    Args:
        pred_embeddings: Predicted embeddings ``(B, D)``, L2-normalised.
        class_embeddings: Class embedding matrix ``(C, D)``, L2-normalised.
        true_label_indices: Ground-truth class indices ``(B,)``, int64.

    Returns:
        Float accuracy in ``[0, 1]`` — fraction of samples where the
        nearest-neighbour class matches the ground truth.
    """
    similarities = pred_embeddings @ class_embeddings.T  # (B, C)
    predicted = similarities.argmax(dim=-1)  # (B,)
    correct = (predicted == true_label_indices).float()
    return correct.mean().item()


def topk_accuracy(
    pred_embeddings: Tensor,
    class_embeddings: Tensor,
    true_label_indices: Tensor,
    k: int = 5,
) -> float:
    """Compute top-k nearest-neighbour accuracy over a batch.

    Args:
        pred_embeddings: Predicted embeddings ``(B, D)``, L2-normalised.
        class_embeddings: Class embedding matrix ``(C, D)``, L2-normalised.
        true_label_indices: Ground-truth class indices ``(B,)``, int64.
        k: Number of nearest classes to consider.

    Returns:
        Float accuracy in ``[0, 1]`` — fraction of samples where the
        ground-truth class appears in the top-k nearest neighbours.
    """
    similarities = pred_embeddings @ class_embeddings.T  # (B, C)
    _, top_indices = similarities.topk(k=k, dim=-1, largest=True)  # (B, k)
    true_expanded = true_label_indices.unsqueeze(1).expand_as(top_indices)
    correct = top_indices.eq(true_expanded).any(dim=1).float()
    return correct.mean().item()


# =============================================================================
# Embedding quality analysis
# =============================================================================


def embedding_coverage(
    vocab: list[str],
    word2vec_bin: str | Path,
) -> dict[str, list[str]]:
    """Check which vocabulary words are missing from a Word2Vec model.

    Args:
        vocab: List of class name strings to check.
        word2vec_bin: Path to the Google News Word2Vec binary file.

    Returns:
        Dict with keys:
            - ``"found"``: Words resolved directly or via token averaging.
            - ``"oov"``: Words with no tokens in the Word2Vec vocabulary.

    Raises:
        FileNotFoundError: If ``word2vec_bin`` does not exist.
    """
    from gensim.models import KeyedVectors  # type: ignore[import]

    word2vec_bin = Path(word2vec_bin)
    if not word2vec_bin.exists():
        raise FileNotFoundError(f"Word2Vec binary not found: {word2vec_bin}")

    log.info("Checking vocabulary coverage in Word2Vec model …")
    wv = KeyedVectors.load_word2vec_format(str(word2vec_bin), binary=True)

    found: list[str] = []
    oov: list[str] = []

    for word in vocab:
        tokens = word.lower().split()
        has_any = any(t in wv for t in tokens)
        (found if has_any else oov).append(word)

    log.info(
        "Coverage: %d/%d words resolved, %d OOV",
        len(found),
        len(vocab),
        len(oov),
    )
    return {"found": found, "oov": oov}


def mean_pairwise_similarity(embeddings: Tensor | np.ndarray) -> float:
    """Compute the mean cosine similarity between all pairs of class embeddings.

    A low value (close to 0) indicates well-separated class embeddings,
    which is desirable for nearest-neighbour retrieval.  A high value suggests
    many semantically similar class pairs that may be prone to confusion.

    Args:
        embeddings: Class embedding matrix ``(C, D)``, L2-normalised.

    Returns:
        Mean pairwise cosine similarity (scalar float, excluding self-pairs).
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings.astype(np.float32))

    C = embeddings.shape[0]
    sim_matrix = embeddings @ embeddings.T  # (C, C)

    # Exclude the diagonal (self-similarity = 1.0)
    mask = ~torch.eye(C, dtype=torch.bool)
    off_diag = sim_matrix[mask]
    return off_diag.mean().item()


def find_confusable_pairs(
    embeddings: Tensor | np.ndarray,
    vocab: list[str],
    threshold: float = 0.7,
    top_n: int = 20,
) -> list[tuple[str, str, float]]:
    """Find class pairs whose Word2Vec embeddings are dangerously similar.

    Pairs above ``threshold`` cosine similarity are flagged as likely
    confusable during nearest-neighbour retrieval.

    Args:
        embeddings: Class embedding matrix ``(C, D)``, L2-normalised.
        vocab: Ordered class name list of length ``C``.
        threshold: Cosine similarity above which a pair is flagged.
        top_n: Maximum number of pairs to return, sorted by descending
            similarity.

    Returns:
        List of ``(word_a, word_b, similarity)`` tuples, sorted by
        descending similarity score.
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings.astype(np.float32))

    C = embeddings.shape[0]
    sim_matrix = (embeddings @ embeddings.T).numpy()

    pairs: list[tuple[str, str, float]] = []
    for i in range(C):
        for j in range(i + 1, C):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                pairs.append((vocab[i], vocab[j], sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]
