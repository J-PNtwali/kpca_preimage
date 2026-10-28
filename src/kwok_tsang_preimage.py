"""
Kwok & Tsang pre-image method.

Reference:
    J.-Y. Kwok and I.-H. Tsang, "The pre-image problem in kernel methods,"
    IEEE Transactions on Neural Networks, 15(6):1517-1525, 2004.

Idea
----
Instead of Mika et al.'s fixed-point iteration (which can be unstable and
needs a good initial guess), Kwok & Tsang exploit a *distance-preservation*
argument:

  1. For an RBF kernel k(x,y) = exp(-gamma * ||x-y||^2), the squared
     distance between two points in feature space is a monotonic function
     of the squared distance between them in input space:

         ||phi(x) - phi(y)||^2 = 2 - 2*k(x,y)
                                = 2*(1 - exp(-gamma * ||x-y||^2))

     This can be inverted:

         ||x-y||^2 = -(1/gamma) * ln(1 - ||phi(x)-phi(y)||^2 / 2)

  2. So, given the target point's location in feature space (or in the
     kernel-PCA-reduced subspace, which is the standard approximation),
     we can estimate its *input-space* squared distance to each of its
     n nearest training neighbours, without ever inverting phi directly.

  3. Once we have n estimated distances to n known input-space points,
     recovering the pre-image is a classic multilateration problem: we
     solve a linear least-squares system (the same trick used in
     trilateration / GPS positioning) for the point x consistent with
     those distances.

This avoids the iterative, locally-convergent fixed-point scheme entirely.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA


def _resolve_gamma(kpca: KernelPCA) -> float:
    """sklearn stores the *resolved* gamma in `gamma_` (post 1.x), falling
    back to `gamma` for older versions where it may already be a float."""
    if hasattr(kpca, "gamma_") and kpca.gamma_ is not None:
        return float(kpca.gamma_)
    if kpca.gamma is not None:
        return float(kpca.gamma)
    # sklearn's default: 1 / n_features
    return 1.0 / kpca.X_fit_.shape[1]


def kwok_tsang_preimage(
    kpca: KernelPCA,
    x_transformed: np.ndarray,
    n: int = 10,
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Recover pre-images for kernel-PCA-reduced points using the
    Kwok & Tsang (2004) distance-based method.

    Parameters
    ----------
    kpca : sklearn.decomposition.KernelPCA
        A fitted KernelPCA object with kernel='rbf'. Must have been fit
        (so that kpca.X_fit_, kpca.eigenvectors_, kpca.eigenvalues_ exist).
    x_transformed : np.ndarray, shape (n_samples, n_components)
        The kernel-PCA coordinates of the point(s) whose pre-image(s) we
        want to recover -- i.e. the output of kpca.transform(...) for
        the target points (possibly denoised / modified in the reduced
        space before calling this function).
    n : int
        Number of nearest neighbours (in the reduced feature space) used
        to set up the local distance-based reconstruction. Must be at
        least d+1 where d is the input dimensionality, for the linear
        system to be well posed; using more neighbours turns the solve
        into a least-squares fit, which is generally more robust.
    ridge : float
        Small ridge regularisation added to the linear system for
        numerical stability when neighbours are near-degenerate.

    Returns
    -------
    np.ndarray, shape (n_samples, n_input_features)
        The estimated pre-images in input space.

    Notes
    -----
    - Implemented for the RBF kernel, as in the original paper. Other
      kernels would need a different feature-space-to-input-space
      distance relationship (e.g. for polynomial kernels, the inverse
      is not available in closed form in general).
    - The "feature space" distances used here are computed in the
      *reduced* kernel PCA subspace (the top n_components directions),
      which is the standard practical approximation: computing exact
      full feature-space distances is not tractable when phi is
      infinite-dimensional.
    """
    if kpca.kernel != "rbf":
        raise NotImplementedError(
            "kwok_tsang_preimage currently only supports kernel='rbf' "
            f"(got kernel={kpca.kernel!r}). The distance-inversion step "
            "relies on the closed-form RBF relationship between feature-"
            "space and input-space distances."
        )

    X_train = kpca.X_fit_                      # (n_train, d)
    n_train, d = X_train.shape
    gamma = _resolve_gamma(kpca)

    # if n < d + 1:
    #     raise ValueError(
    #         f"n={n} neighbours is too few for a well-posed solve in "
    #         f"d={d} input dimensions; use n >= d+1 (ideally n > d+1 for "
    #         "a more robust least-squares fit)."
    #     )
    if n > n_train:
        raise ValueError(f"n={n} exceeds the number of training points ({n_train}).")

    # Reduced-space coordinates of all training points (their own
    # kernel PCA embedding) -- used to find neighbours of each target.
    Z_train = kpca.transform(X_train)           # (n_train, n_components)

    x_transformed = np.atleast_2d(x_transformed)
    preimages = np.zeros((x_transformed.shape[0], d))

    # Precompute squared norms of training points for the linear system
    train_sq_norms = np.sum(X_train ** 2, axis=1)   # (n_train,)

    for row, z in enumerate(x_transformed):
        # --- Step 1: find n nearest neighbours in the reduced feature space
        d_feat_sq = np.sum((Z_train - z) ** 2, axis=1)      # (n_train,)
        nn_idx = np.argsort(d_feat_sq)[:n]
        d_feat_sq_nn = d_feat_sq[nn_idx]
        X_nn = X_train[nn_idx]                              # (n, d)
        nn_sq_norms = train_sq_norms[nn_idx]                 # (n,)

        # --- Step 2: convert feature-space squared distances to estimated
        # input-space squared distances via the RBF inversion formula.
        # Clip to avoid log(<=0) from numerical noise (d_feat_sq should be
        # in [0, 2] since it derives from a kernel value in [0, 1]).
        ratio = np.clip(1.0 - d_feat_sq_nn / 2.0, 1e-12, 1.0)
        d_input_sq = -np.log(ratio) / gamma                  # (n,)

        # --- Step 3: solve the multilateration linear system.
        # Using neighbour 0 as the reference point, subtract its equation
        # from all others to eliminate the quadratic ||x||^2 term:
        #   ||x - x_i||^2 = d_i^2   for i = 0..n-1
        #   =>  2 x^T (x_0 - x_i) = d_i^2 - d_0^2 - ||x_i||^2 + ||x_0||^2
        # for i = 1..n-1. This gives an (n-1) x d linear system in x.
        A = 2.0 * (X_nn[0] - X_nn[1:])                       # (n-1, d)
        b = (
            d_input_sq[1:] - d_input_sq[0]
            - nn_sq_norms[1:] + nn_sq_norms[0]
        )                                                     # (n-1,)

        # Ridge-regularised least squares (robust to near-collinear
        # neighbours, e.g. when n-1 < d or neighbours are nearly coplanar).
        ATA = A.T @ A + ridge * np.eye(d)
        ATb = A.T @ b
        x_hat = np.linalg.solve(ATA, ATb)

        preimages[row] = x_hat

    return preimages