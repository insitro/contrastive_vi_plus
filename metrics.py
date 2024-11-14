import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors


def entropy_batch_mixing(
    data, labels, n_neighbors=50, n_pools=50, n_samples_per_pool=100
):
    """Computes Entory of Batch mixing metric for ``adata`` given the batch column name.
    Parameters
    ----------
    data
        Numpy ndarray of data
    labels
        Numpy ndarray of labels
    n_neighbors: int
        Number of nearest neighbors.
    n_pools: int
        Number of EBM computation which will be averaged.
    n_samples_per_pool: int
        Number of samples to be used in each pool of execution.
    Returns
    -------
    score: float
        EBM score. A float between zero and one.
    """

    def __entropy_from_indices(indices, n_cat):
        return entropy(
            np.array(np.unique(indices, return_counts=True)[1].astype(np.int32)),
            base=n_cat,
        )

    n_cat = len(np.unique(labels))

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(data)
    indices = neighbors.kneighbors(data, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: labels[i])(indices)

    entropies = np.apply_along_axis(
        __entropy_from_indices, axis=1, arr=batch_indices, n_cat=n_cat
    )

    # average n_pools entropy results where each result is
    # an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean(
            [
                np.mean(
                    entropies[np.random.choice(len(entropies), size=n_samples_per_pool)]
                )
                for _ in range(n_pools)
            ]
        )

    return score
