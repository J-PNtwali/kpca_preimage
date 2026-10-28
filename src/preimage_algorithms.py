from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel

def calculate_gammas(kpca_obj, x_transformed):
    # properly scaled eigenvectors
    non_zeros = np.flatnonzero(kpca_obj.eigenvalues_)
    scaled_alphas = np.zeros_like(kpca_obj.eigenvectors_)
    scaled_alphas[:, non_zeros] = kpca_obj.eigenvectors_[:, non_zeros] / np.sqrt(kpca_obj.eigenvalues_[non_zeros])

    # calculate gammas
    gammas_tilde = np.sum(scaled_alphas * x_transformed, axis=1, keepdims=True)
    gammas = gammas_tilde + (1 - np.sum(gammas_tilde))/gammas_tilde.shape[0]
    
    return gammas

def mika_etal_preimage(kpca_obj, x_transformed, x0, x_ref=None, lambda_=0.0, tol=1e-9, max_iter=1000, verbose=False):
    r''' 
    Implementation of the kernel-PCA preimage method proposed in the papers below. 
    This correponds to the method of [1] when lambda_=0 and [2] otherwise.

    [1] Mika, Sebastian, et al. "Kernel PCA and de-noising in feature spaces." Advances in neural information processing systems 11 (1998).

    [2] Abrahamsen, Trine Julie, and Lars Kai Hansen. "Input space regularization stabilizes pre-images for kernel PCA de-noising." 2009 IEEE International Workshop on Machine Learning for Signal Processing. IEEE, 2009.

    Inputs:
    -------
    - kpca_obj: a fitted kernel PCA object

    - x_transformed: KPCA projection, i.e. P_n \phi(x) for some point x

    - x0: initial point for iteration scheme

    - x_ref: noisy observation needed when lambda_ > 0 (default: None)

    - lambda_: regularization parameter

    Outputs:
    --------
    - x1: approximate preimage 

    - iter: number of iterations. iter = max_iter indicates that the method did not converge
    '''
    if lambda_ > 0:
        assert x_ref is not None, 'x_ref must be provided when lambda_ > 0'
    else:
        x_ref = np.zeros_like(x0)
    
    # calculate the multiplying factor
    if kpca_obj.kernel == 'rbf':
        factor = 2 * kpca_obj.gamma if lambda_ > 0 else 1
    else:
       raise NotImplementedError('Preimage method not implemented for the specified kernel')
    # calculate the gammas
    gammas = calculate_gammas(kpca_obj, x_transformed)

    stop = 0
    for iter in range(max_iter):
        if stop:
            break
        else:
            # calculate coeffs
            Kx0 = kpca_obj._get_kernel(kpca_obj.X_fit_, x0)
            
            weights = Kx0 * gammas

            # calculate z1 (z_{i+1})
            num = factor * np.sum(kpca_obj.X_fit_ * weights, axis=0, keepdims=True) + lambda_ * x_ref
            den = factor * np.sum(weights) + lambda_
           
            x1 = num / den

            # check stopping criteria
            normdiff = np.linalg.norm(x1 - x0)
            if  normdiff < tol:
                stop = 1
            else:
                x0 = x1.copy()
            
            if verbose:
                print(f'iter {iter+1:<4} | norm-diff {normdiff:.1e}')
    if iter == max_iter-1:
        print('Reached max_iter without convergence!')
    return x1, iter+1


def kwok_tsang_preimage(
    kpca: KernelPCA,
    x_transformed: np.ndarray,
    n: int = 10,
) -> np.ndarray:
    """
    Kwok & Tsang (2004) distance-constraint pre-image method
    for an RBF-kernel KernelPCA model.

    Parameters
    ----------
    kpca : sklearn.decomposition.KernelPCA
        A fitted KernelPCA model with kernel='rbf'.

    x_transformed : ndarray, shape (n_components,) or
                    (n_targets, n_components)
        Target KPCA coordinates, as returned by kpca.transform().
        Coordinates may also have been modified in KPCA space,
        e.g. for denoising.

    n : int, default=10
        Number of nearest training samples in feature space.
        Must satisfy 2 <= n <= n_train. Unlike a full-dimensional
        trilateration solver, this method does not require n >= d+1.

    Returns
    -------
    preimages : ndarray, shape (n_targets, n_features)
        Approximate pre-images in the original input space.

    Notes
    -----
    Implements the RBF-kernel version of the distance-based method
    in Kwok & Tsang, IEEE TNN, 15(6), 1517-1525, 2004.

    No ridge penalty is added: the reconstruction follows the
    paper's local SVD/least-squares formulation.

    The inversion from feature-space to input-space distances is
    approximate because a projected KPCA point need not itself
    be an exact image of any input point under the RBF feature map.
    """
    # ------------------------------------------------------------
    # 0. Validate the fitted model and inputs
    # ------------------------------------------------------------
    if getattr(kpca, "kernel", None) != "rbf":
        raise NotImplementedError(
            "This implementation supports only kernel='rbf'."
        )

    if not hasattr(kpca, "X_fit_"):
        raise ValueError("kpca must be fitted before calling this function.")

    X_fit_original = kpca.X_fit_
    X_train = np.asarray(X_fit_original, dtype=float)

    if X_train.ndim != 2 or X_train.shape[0] == 0:
        raise ValueError("The fitted training data must be a nonempty 2D array.")

    n_train, d = X_train.shape

    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be an integer.")
    if not 2 <= n <= n_train:
        raise ValueError(f"n must satisfy 2 <= n <= {n_train}.")

    targets = np.asarray(x_transformed, dtype=float)
    if targets.ndim == 1:
        targets = targets[np.newaxis, :]
    if targets.ndim != 2:
        raise ValueError("x_transformed must be a 1D or 2D array.")
    if not np.isfinite(targets).all():
        raise ValueError("x_transformed contains non-finite values.")

    # Resolve gamma consistently with scikit-learn's RBF kernel.
    gamma = getattr(kpca, "gamma_", None)
    if gamma is None:
        gamma = kpca.gamma
    if gamma is None:
        gamma = 1.0 / d
    gamma = float(gamma)

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("The fitted RBF gamma must be finite and positive.")

    # ------------------------------------------------------------
    # 1. Compute the training scores and full centered kernel norms
    # ------------------------------------------------------------
    # These coordinates must come from the same fitted KPCA model
    # as x_transformed.
    Z_train = np.asarray(
        kpca.transform(X_fit_original),
        dtype=float,
    )

    if targets.shape[1] != Z_train.shape[1]:
        raise ValueError(
            f"Expected {Z_train.shape[1]} KPCA coordinates per target, "
            f"got {targets.shape[1]}."
        )

    # Full, uncentered RBF Gram matrix on the training observations.
    K = rbf_kernel(X_train, X_train, gamma=gamma)

    # Diagonal of the centered Gram matrix:
    # ||phi(x_i) - mean(phi(X_train))||^2.
    row_means = K.mean(axis=1)
    grand_mean = K.mean()
    centered_feature_norm_sq = (
        np.diag(K) - 2.0 * row_means + grand_mean
    )

    # Protect against tiny negative values due to floating-point error.
    centered_feature_norm_sq = np.maximum(
        centered_feature_norm_sq, 0.0
    )

    # ------------------------------------------------------------
    # 2. Process each target
    # ------------------------------------------------------------
    preimages = np.empty((targets.shape[0], d), dtype=float)

    for i, z in enumerate(targets):
        # --------------------------------------------------------
        # Step A: feature-space distances (paper, Section III-A)
        # --------------------------------------------------------
        # ||P phi(x) - phi(x_j)||^2
        #   = ||z||^2
        #     + ||phi(x_j) - mean(phi(X_train))||^2
        #     - 2 * <z, centered_score(x_j)>.
        #
        # The centered_feature_norm_sq term retains the contribution
        # of feature-space directions omitted by the KPCA projection.
        d_feature_sq = (
            np.dot(z, z)
            + centered_feature_norm_sq
            - 2.0 * (Z_train @ z)
        )

        # Only correct tiny numerical negatives; exact squared
        # distances are nonnegative.
        d_feature_sq = np.maximum(d_feature_sq, 0.0)

        # The paper selects neighbors using feature-space distances.
        nn_idx = np.argsort(d_feature_sq, kind="stable")[:n]

        X_nn = X_train[nn_idx]
        d_feature_sq_nn = d_feature_sq[nn_idx]

        # --------------------------------------------------------
        # Step B: invert the RBF distance relationship
        # (paper, Section III-B, Eq. 11)
        # --------------------------------------------------------
        # For k(x,y) = exp(-gamma * ||x-y||^2):
        #
        # d_input^2 = -log(1 - d_feature^2 / 2) / gamma.
        #
        # This is the paper's approximation for the projected target.
        implied_kernel = 1.0 - 0.5 * d_feature_sq_nn

        if np.any(implied_kernel <= 0.0):
            raise ValueError(
                f"Target {i}: at least one selected neighbor has "
                "1 - d_feature_sq/2 <= 0, so its RBF distance "
                "cannot be inverted to a finite real input-space "
                "distance. The projected target may be too far from "
                "the training feature vectors for this approximation."
            )

        d_input_sq = -np.log(implied_kernel) / gamma

        # --------------------------------------------------------
        # Step C: center the neighbors and compute their SVD
        # (paper, Section III-C)
        # --------------------------------------------------------
        x_bar = X_nn.mean(axis=0)
        X_centered = X_nn - x_bar  # shape (n, d)

        # X_centered = U @ diag(s) @ Vt.
        # The columns of Vt.T span the selected neighbors' input
        # directions around their centroid.
        _, singular_values, Vt = np.linalg.svd(
            X_centered,
            full_matrices=False,
        )

        if singular_values.size == 0 or singular_values[0] == 0.0:
            raise ValueError(
                f"Target {i}: the selected neighbors have zero "
                "centered rank; their geometry cannot determine "
                "a pre-image."
            )

        # Numerical rank, following the usual SVD tolerance.
        svd_tol = (
            np.finfo(float).eps
            * max(X_centered.shape)
            * singular_values[0]
        )
        rank = int(np.sum(singular_values > svd_tol))

        if rank == 0:
            raise ValueError(
                f"Target {i}: the selected neighbors are numerically "
                "rank-deficient (rank zero)."
            )

        # Orthonormal basis for the neighbors' centered input span.
        basis = Vt[:rank].T                  # (d, rank)
        coords = X_centered @ basis          # (n, rank)

        # Squared distances of the centered neighbors to their centroid.
        d0_sq = np.sum(coords**2, axis=1)

        # From the paper's distance constraints, define
        # y_j = d_input_j^2 - ||coords_j||^2.
        #
        # Centering y eliminates the unknown ||z_hat||^2 term.
        y = d_input_sq - d0_sq
        y -= y.mean()

        # Least-squares solution of:
        #     -2 * coords @ z_hat ~= y
        #
        # lstsq avoids explicitly forming and inverting a normal
        # equation matrix.
        z_hat = np.linalg.lstsq(
            coords,
            -0.5 * y,
            rcond=None,
        )[0]

        # Transform back to the original input coordinates.
        preimages[i] = x_bar + basis @ z_hat

    return preimages


def honeine_rich_preimage(
    kpca: KernelPCA,
    x_transformed: np.ndarray,
    lambda_: float = 1e-9,
    include_feature_mean: bool = True,
    kernel_jitter: float = 0.0,
    rcond: float | None = None,
) -> np.ndarray:
    """
    Honeine & Richard (2011) conformal-map / inverse-map
    pre-image method for kernel PCA.
    https://link.springer.com/content/pdf/10.1007/s11265-010-0482-9.pdf

    Parameters
    ----------
    kpca : sklearn.decomposition.KernelPCA
        A fitted KernelPCA estimator. The original input-space
        training samples must be available in kpca.X_fit_.
        Standard and callable kernels are supported.

    x_transformed : ndarray, shape (n_components,) or
                    (n_targets, n_components)
        Target KPCA scores, normally from kpca.transform().
        Modified/denoised scores are also accepted.

    lambda_ : float, default=1e-9
        Regularization parameter in the Honeine-Richard
        inverse-map formulation. Must be nonnegative.

    include_feature_mean : bool, default=True
        If True, add the feature-space training mean to the
        reconstructed target, as in standard KPCA reconstruction
        from centered scores.

        Set False when the intended target is only the projection
        onto the retained centered principal subspace.

    kernel_jitter : float, default=0.0
        Optional diagonal jitter added to the uncentered training
        kernel Gram matrix before solving for K^{-1} gamma.
        This is a numerical stabilization, not part of the paper's
        original equations. Normally leave it at zero.

    rcond : float or None, default=None
        Cutoff passed to np.linalg.lstsq for the final input-space
        least-squares solution.

    Returns
    -------
    ndarray, shape (n_targets, n_features)
        Estimated pre-images in input space.

    Notes
    -----
    The paper's inverse-map equation is

        X.T @ x_star ~= (P - lambda_ * K^{-1}) @ gamma,

    where P = X @ X.T is the input-space Gram matrix and K is
    the uncentered kernel Gram matrix.

    The implementation solves the corresponding least-squares
    problem directly, without explicitly inverting P or K.
    """
    # ----------------------------------------------------------
    # 0. Validate the fitted estimator and inputs
    # ----------------------------------------------------------
    if not isinstance(kpca, KernelPCA):
        raise TypeError("kpca must be a fitted KernelPCA estimator.")

    for attr in ("X_fit_", "eigenvectors_", "eigenvalues_"):
        if not hasattr(kpca, attr):
            raise ValueError("kpca must be fitted before use.")

    if kpca.kernel == "precomputed":
        raise ValueError(
            "kernel='precomputed' is unsupported because the "
            "original input-space training observations are required."
        )

    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and nonnegative.")

    if not np.isfinite(kernel_jitter) or kernel_jitter < 0:
        raise ValueError(
            "kernel_jitter must be finite and nonnegative."
        )

    X_fit = kpca.X_fit_
    if hasattr(X_fit, "toarray"):
        X_fit = X_fit.toarray()

    X = np.asarray(X_fit, dtype=float)

    n_train, n_features = X.shape

    scores = np.asarray(x_transformed, dtype=float)
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]

    V = np.asarray(kpca.eigenvectors_, dtype=float)
    eigenvalues = np.asarray(kpca.eigenvalues_, dtype=float)

    # ----------------------------------------------------------
    # 1. Convert KPCA scores into kernel-expansion coefficients
    # ----------------------------------------------------------
    # A[:, j] = eigenvectors_[:, j] / sqrt(eigenvalues_[j]).
    # These are the coefficients of normalized principal
    # directions in the original, uncentered kernel basis.
    A = np.zeros_like(V)
    positive = eigenvalues > 0

    A[:, positive] = (
        V[:, positive] / np.sqrt(eigenvalues[positive])[np.newaxis, :]
    )

    # Gamma[t, i] is coefficient gamma_i for target t.
    Gamma = scores @ A.T

    # Add the RKHS feature mean, mu = (1/N) sum_i phi(x_i).
    # This represents mu + the reconstruction from centered scores.
    if include_feature_mean:
        Gamma += 1.0 / n_train

    # ----------------------------------------------------------
    # 2. Build the input-space Gram matrix P = X X.T
    # ----------------------------------------------------------
    P = X @ X.T

    # Right-hand side corresponding to P @ gamma.
    # Row-wise representation for all targets:
    #     B = Gamma @ P - lambda_ * (K^{-1} Gamma.T).T
    rhs = Gamma @ P

    # ----------------------------------------------------------
    # 3. Apply the Honeine-Richard inverse-map correction
    # ----------------------------------------------------------
    if lambda_ > 0:
        # UNCENTERED kernel Gram matrix.
        K = np.asarray(kpca._get_kernel(X, X), dtype=float)

        if kernel_jitter > 0:
            K = K + kernel_jitter * np.eye(n_train)

        try:
            KinvGammaT = np.linalg.solve(K, Gamma.T)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "The training kernel Gram matrix is singular. "
                "Check duplicate samples and kernel conditioning. "
                "A small positive kernel_jitter may help numerically, "
                "but changes the exact paper equation."
            ) from exc

        rhs -= lambda_ * KinvGammaT.T

    # ----------------------------------------------------------
    # 4. Solve the input-space least-squares problem
    # ----------------------------------------------------------
    # For each target, solve X @ x_star ~= rhs[t].
    # This is Eq. (8) in the paper, in row-oriented notation.
    preimages = np.linalg.lstsq(
        X,
        rhs.T,
        rcond=rcond,
    )[0].T

    return preimages