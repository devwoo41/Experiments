"""Agent 2 — Topic Mining & Clustering (paper §2.2.2, K.2).

Faithful to the paper:
  - Embeddings via sentence-transformers `all-MiniLM-L6-v2` on title ⊕ abstract.
  - K-means with K* = argmax silhouette over K ∈ [5, 15].
  - TF-IDF on titles+abstracts -> top terms become the cluster name.
  - Reports silhouette, Calinski-Harabasz, Davies-Bouldin metrics.
  - Computes per-paper cluster confidence and flags outliers.
  - Inter-cluster relationship strength = cosine(centroid_i, centroid_j).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ..llm import GeminiLLM
from ..state import (
    Cluster,
    ClusterQualityMetrics,
    ClusterRelationship,
    Paper,
    SurveyState,
)


_STOP_WORDS = "english"


def _encode(texts: list[str], model_name: str) -> np.ndarray:
    """Lazily load sentence-transformers to avoid heavy import at module load."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(emb, dtype=np.float32)


def _select_k(matrix: np.ndarray, k_min: int, k_max: int) -> tuple[int, dict[int, float]]:
    """Pick K maximizing silhouette over [k_min, k_max] (paper §2.2.2)."""
    n = matrix.shape[0]
    k_max = min(k_max, n - 1)
    k_min = min(k_min, k_max)
    if k_min < 2:
        k_min = 2
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(matrix, labels, metric="euclidean"))
    if not scores:
        return max(2, min(k_max, 5)), {}
    best_k = max(scores, key=scores.get)
    return best_k, scores


def _tfidf_cluster_names(papers: list[Paper], labels: np.ndarray, top_n: int
                         ) -> dict[int, list[str]]:
    """Top-N TF-IDF terms per cluster, using paper §2.2.2 formula verbatim:

        TF-IDF(w, Cj) = TF(w, Cj) * log( K / |{Ck : w in Ck}| )

    Each cluster is treated as a SINGLE document (concatenation of its papers'
    title+abstract). DF counts the number of clusters in which the term appears.
    """
    from collections import defaultdict

    cluster_docs: dict[int, list[str]] = defaultdict(list)
    for i, l in enumerate(labels):
        cluster_docs[int(l)].append(
            f"{papers[i].get('title','')} {papers[i].get('abstract','')}"
        )
    cluster_ids = sorted(cluster_docs.keys())
    cluster_texts = [" ".join(cluster_docs[c]) for c in cluster_ids]
    K = len(cluster_ids)

    # TF per cluster (cluster = document)
    cv = CountVectorizer(stop_words=_STOP_WORDS, max_features=5000,
                         ngram_range=(1, 2))
    tf = cv.fit_transform(cluster_texts).toarray().astype(np.float64)  # (K, V)
    vocab = np.array(cv.get_feature_names_out())

    # DF (paper formula): #clusters containing the term
    df = (tf > 0).sum(axis=0)
    df_safe = np.maximum(df, 1)
    idf = np.log(K / df_safe)            # paper-exact, no smoothing
    scores = tf * idf                    # broadcast (K, V)

    out: dict[int, list[str]] = {}
    for ci, cid in enumerate(cluster_ids):
        row = scores[ci]
        top = np.argsort(row)[::-1][:top_n]
        out[cid] = [vocab[i] for i in top if row[i] > 0]
    return out


def _confidence(matrix: np.ndarray, labels: np.ndarray,
                centroids: np.ndarray) -> np.ndarray:
    """Cluster confidence = 1 - d(x, own_centroid) / max_k d(x, centroid_k).

    Direct implementation of the formula in paper §2.2.2.
    """
    n = matrix.shape[0]
    conf = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d_all = np.linalg.norm(centroids - matrix[i], axis=1)
        own = d_all[int(labels[i])]
        denom = float(d_all.max()) or 1e-9
        conf[i] = 1.0 - own / denom
    return conf


def cluster_node(state: SurveyState, llm: GeminiLLM) -> dict[str, Any]:
    cfg = state["config"]["clustering"]
    papers: list[Paper] = state["papers"]

    if len(papers) < cfg["k_min"]:
        return {
            "clusters": [], "cluster_relationships": [],
            "cluster_quality_metrics": {}, "outliers": [],
            "logs": state.get("logs", []) + [
                f"[Cluster] only {len(papers)} papers, < k_min={cfg['k_min']}; skipping clustering"
            ],
        }

    texts = [f"{p.get('title','')}. {p.get('abstract','')}" for p in papers]
    matrix = _encode(texts, cfg["embedding_model"])

    best_k, k_scores = _select_k(matrix, cfg["k_min"], cfg["k_max"])
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(matrix)
    centroids = km.cluster_centers_

    silhouette = float(silhouette_score(matrix, labels)) if len(set(labels)) >= 2 else 0.0
    ch = float(calinski_harabasz_score(matrix, labels)) if len(set(labels)) >= 2 else 0.0
    db = float(davies_bouldin_score(matrix, labels)) if len(set(labels)) >= 2 else 0.0
    metrics: ClusterQualityMetrics = {
        "k_selected": int(best_k),
        "silhouette": silhouette,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "k_candidates": k_scores,
    }

    confidences = _confidence(matrix, labels, centroids)
    outlier_threshold = cfg["outlier_confidence_threshold"]
    outliers: list[str] = []
    for i, p in enumerate(papers):
        p["cluster_id"] = int(labels[i])
        p["cluster_confidence"] = float(confidences[i])
        if confidences[i] < outlier_threshold:
            outliers.append(p["paper_id"])

    names = _tfidf_cluster_names(papers, labels, top_n=cfg["tfidf_top_terms"])

    clusters: list[Cluster] = []
    for cid in sorted(set(int(l) for l in labels)):
        member_ids = [papers[i]["paper_id"] for i in range(len(papers)) if int(labels[i]) == cid]
        key_terms = names.get(cid, [])
        name = " ".join(key_terms[:3]).title() if key_terms else f"Cluster {cid}"
        clusters.append(Cluster(
            cluster_id=int(cid),
            name=name,
            key_terms=key_terms,
            paper_ids=member_ids,
            size=len(member_ids),
        ))

    # Inter-cluster cosine-similarity relationships
    rels: list[ClusterRelationship] = []
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    norm_centroids = centroids / norms
    K = centroids.shape[0]
    for i in range(K):
        for j in range(i + 1, K):
            sim = float(np.dot(norm_centroids[i], norm_centroids[j]))
            rels.append(ClusterRelationship(a=int(i), b=int(j), strength=sim))

    log = (
        f"[Cluster] K*={best_k} (silhouette={silhouette:.3f}, "
        f"CH={ch:.1f}, DB={db:.3f}), {len(outliers)} outliers"
    )
    return {
        "clusters": clusters,
        "cluster_relationships": rels,
        "cluster_quality_metrics": metrics,
        "outliers": outliers,
        "papers": papers,
        "logs": state.get("logs", []) + [log],
    }
