from typing import Dict, Callable, Optional, List, Any, Union

import numpy as np
from sklearn.neighbors import KernelDensity


def mean_of_mean_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.mean(axis=1).mean()
    return s


def mean_of_mean_influence_success(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.mean(axis=1).mean() if is_success else 0.0
    return s


def sum_of_sum_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.sum(axis=1).sum()
    return s


def sum_of_sum_influence_success(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.sum(axis=1).sum() if is_success else 0.0
    return s


def min_of_max_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.max(axis=1).min()
    return -s if is_success else s


def max_of_min_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.min(axis=1).max()
    return s if is_success else -s


def batch_kde_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str = "gaussian",
    bandwidth: Union[float, str] = 1.0,
) -> np.ndarray:
    """Evaluate the log-likelihood of a batch of test points under their corresponding KDEs.

    Args:
        x: (B, N, D) tensor.
        y: (B, D) array.
        kernel: Kernel type for KDE.
        bandwidth: Bandwidth for KDE, or 'max_eig' to compute adaptive bandwidth.

    Returns:
        log_likelihoods: (B,) array of log-likelihoods for each sample in y.
    """
    assert x.ndim == 3
    batch_size = x.shape[0]
    sample_dim = x.shape[2]

    assert y.ndim == 2 and y.shape == (batch_size, sample_dim)
    
    log_likelihoods = np.zeros(batch_size)
    for i in range(batch_size):
        x_i = x[i]
        y_i = y[i:i+1]

        bw = bandwidth
        if isinstance(bandwidth, str):
            if bandwidth == "max_eig":
                cov = np.cov(x_i.T)
                max_eig = np.max(np.linalg.eigvalsh(cov))
                bw = np.sqrt(max_eig)
            else:
                raise ValueError(f"Bandwidth {bandwidth} is not supported.")

        kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(x_i)
        log_likelihoods[i] = kde.score_samples(y_i)[0]

    return log_likelihoods


def compute_l2_distance(
    x: np.ndarray,
    y: np.ndarray,
    normalization: Optional[str] = "uniform"
) -> np.ndarray:
    """Compute L2-norm between all pairs of vectors.

    Args:
        x: (N, D) matrix.
        y: (M, D) matrix.
        normalization: Type of normalization to apply before computing L2 distances.

    Returns:
        (N, M) matrix of L2 distances.
    """
    assert x.ndim == 2 and y.ndim == 2

    if normalization == "uniform":
        # Scaling to [0, 1].
        min_val = np.minimum(x.min(axis=0), y.min(axis=0))
        max_val = np.maximum(x.max(axis=0), y.max(axis=0))
        range_val = max_val - min_val + 1e-12
        x = (x - min_val) / range_val
        y = (y - min_val) / range_val

    elif normalization == "gaussian":
        # Z-score standardization.
        mean = np.mean(np.vstack([x, y]), axis=0)
        std = np.std(np.vstack([x, y]), axis=0) + 1e-12
        x = (x - mean) / std
        y = (y - mean) / std

    elif normalization == "unit":
        # Normalize vectors to unit length.
        x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        y_norm = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
        x = x / x_norm
        y = y / y_norm

    # Compute squared L2 distance in memory-efficient expanded form.
    x_norm_sq = np.sum(x**2, axis=1)[:, None]  # (N, 1)
    y_norm_sq = np.sum(y**2, axis=1)[None, :]  # (1, M)
    cross_term = 2 * np.dot(x, y.T)            # (N, M)
    
    return np.sqrt(np.maximum(x_norm_sq + y_norm_sq - cross_term, 0))


def compute_mahal_distance(
    x: np.ndarray, 
    y: np.ndarray
) -> np.ndarray:
    """Compute pairwise Mahalanobis distances between all pairs of vectors."""
    raise NotImplementedError("Results in OOM failures for large tensors x and y.")
    assert x.ndim == 2 and y.ndim == 2

    # Compute inverse covariance matrix.
    cov = np.cov(y.T) + np.eye(y.shape[1]) * 1e-12
    inv_cov = np.linalg.inv(cov)

    # Center data about y mean.
    mean = np.mean(y, axis=0)
    x_centered = x - mean  # (N, D)
    y_centered = y - mean  # (M, D)

    # Compute quadratic Mahalanobis form using matrix multiplication.
    dists = np.zeros((x.shape[0], y.shape[0]))                           # (N, M)
    left = np.dot(x_centered, inv_cov)                                   # (N, D) x (D, D) -> (N, D)
    dists += np.sum(left * x_centered, axis=1)[:, None]                  # (N, 1) -> (N, M)
    dists += np.sum(y_centered @ inv_cov * y_centered, axis=1)[None, :]  # (1, M) -> (N, M)
    dists -= 2 * np.dot(left, y_centered.T)                              # (N, M)

    return np.sqrt(np.maximum(dists, 0))


def compute_cosine_error(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between all pairs of vectors.

    Args:
        x: (N, D) matrix.
        y: (M, D) matrix.

    Returns:
        (N, M) matrix of cosine errors.
    """
    assert x.ndim == 2 and y.ndim == 2

    x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    y_norm = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    return -1.0 * (x / x_norm) @ (y / y_norm).T


EMBEDDING_ERROR_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "l2": compute_l2_distance,
    "cosine": compute_cosine_error,
    "mahal": compute_mahal_distance,
}


def topk_embedding_score(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    error_fn: str = "l2",
    k: int = 5,
    leave_one_out: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Compute top-k embedding similarity scores."""
    scores = EMBEDDING_ERROR_FNS[error_fn](
        test_embeddings,
        data_embeddings,
        **kwargs,
    )
    assert scores.shape == (len(test_embeddings), len(data_embeddings))

    if leave_one_out:
        assert data_embeddings.shape == test_embeddings.shape
        scores += np.diag([np.inf] * data_embeddings.shape[0])

    # Use np.partition to efficiently get the top-k smallest elements.
    partitioned_scores = np.partition(scores, k, axis=1)[:, :k]
    scores = np.mean(partitioned_scores, axis=1)

    return scores


def mahal_embedding_score(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis embedding similarity scores."""

    # Compute inverse covariance matrix.
    cov = np.cov(data_embeddings.T) + np.eye(data_embeddings.shape[1]) * 1e-12
    invcov = np.linalg.inv(cov)
    
    # Compute Mahalanobis distances.
    mean = np.mean(data_embeddings, axis=0)
    mahals = (test_embeddings - mean) @ invcov
    mahals = np.sum((test_embeddings - mean) * mahals, axis=1)
    
    return np.sqrt(np.maximum(mahals, 0))


def compute_test_embedding_scores(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    method: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compute embedding scores for test data."""
    assert data_embeddings.ndim == 2 and test_embeddings.ndim == 2
    
    if method == "topk":
        assert method_kwargs is not None
        scores = topk_embedding_score(
            data_embeddings=data_embeddings,
            test_embeddings=test_embeddings,
            **method_kwargs,
        )
    elif method == "mahal":
        scores = mahal_embedding_score(
            data_embeddings=data_embeddings,
            test_embeddings=test_embeddings,
        )
    else:
        raise ValueError(f"Embedding score method {method} is not supported.")

    assert len(scores) == len(test_embeddings)
    return scores


def sample_to_trajectory_scores(
    sample_scores: np.ndarray,
    num_eps: int,
    ep_idxs: List[np.ndarray],
    ep_lens: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float] = np.mean,
    return_dtype: type = np.float32,
) -> np.ndarray:
    """Aggregate sample scores into trajectory scores via aggr_fn."""
    assert len(sample_scores) == ep_lens.sum()

    traj_scores = np.zeros(num_eps, dtype=return_dtype)
    for i, ep_idx, ep_len in zip(range(num_eps), ep_idxs, ep_lens):
        sample_scores_i: np.ndarray = sample_scores[ep_idx]
        assert sample_scores_i.shape == (ep_len,)
        traj_scores[i] = aggr_fn(sample_scores_i)

    return traj_scores


def pairwise_sample_to_trajectory_scores(
    pairwise_sample_scores: np.ndarray,
    num_test_eps: int,
    num_train_eps: int,
    test_ep_idxs: List[np.ndarray],
    train_ep_idxs: List[np.ndarray],
    test_ep_lens: np.ndarray,
    train_ep_lens: np.ndarray,
    success_mask: np.ndarray,
    aggr_fn: Callable[[np.ndarray, bool], float] = mean_of_mean_influence,
    return_dtype: type = np.float32,
) -> np.ndarray:
    """Aggregate pairwise sample scores into trajectory scores via aggr_fn."""
    assert pairwise_sample_scores.shape == (test_ep_lens.sum(), train_ep_lens.sum())

    traj_scores = np.zeros((num_test_eps, num_train_eps), dtype=return_dtype)
    for i, test_idx in enumerate(test_ep_idxs):
        for j, train_idx in enumerate(train_ep_idxs):
            sample_scores_ij: np.ndarray = pairwise_sample_scores[np.ix_(test_idx, train_idx)]
            assert sample_scores_ij.shape == (test_ep_lens[i], train_ep_lens[j])
            traj_scores[i, j] = aggr_fn(sample_scores_ij, success_mask[i])

    return traj_scores


def compute_demo_quality_scores(
    traj_scores: np.ndarray, 
    success_mask: np.ndarray, 
    metric: str = "net",
) -> Optional[np.ndarray]:
    """Compute demonstration quality scores. Higher predicted value means higher quality."""
    assert traj_scores.shape[0] == len(success_mask)

    has_succ = np.any(success_mask == True)
    has_fail = np.any(success_mask == False)
    
    demo_quality_scores = None
    if metric == "net":
        succ_scores = compute_demo_quality_scores(
            traj_scores=traj_scores, 
            success_mask=success_mask, 
            metric="succ"
        )
        fail_scores = compute_demo_quality_scores(
            traj_scores=traj_scores, 
            success_mask=success_mask, 
            metric="fail"
        )
        succ_scores = succ_scores if succ_scores is not None else 0.0
        fail_scores = fail_scores if fail_scores is not None else 0.0
        demo_quality_scores = succ_scores + fail_scores

    elif metric == "succ":
        demo_quality_scores = traj_scores[success_mask].sum(axis=0) if has_succ else None

    elif metric == "fail":    
        demo_quality_scores = -traj_scores[~success_mask].sum(axis=0) if has_fail else None
        
    else:
        raise ValueError(f"Quality metric {metric} is not supported.")
    
    return demo_quality_scores