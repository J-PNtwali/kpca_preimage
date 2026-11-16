# Authors: 
# Last edit: 2026-04-27

import numpy as np
from sklearn.decomposition import KernelPCA as SklearnKernelPCA

# Standard kernel PCA  inherited from sklearn's KernelPCA with the following modifications:
#---
# (1) we add the proposed inverse transform method: ``inverse_transform_proposed``
#---
# (2) As we need to tune hyperparameters, we modify the existing inverse_transform method to include the ridge regularization parameter at inference time,
#        which allows us to avoid refitting the model with different alpha values, which is computationally expensive.
#        This is implemented in the method ``inverse_transform_kernel_ridge``.
#---
# (3)[TO-DO] We modify the fit method to take ``n_components`` as an argument.
#     This allows us to fit the model with a large number of components and then select the desired number of components at inference time,
#      which is more efficient than refitting the model with different n_components values.
#---
# Note: We use the same principles for the RFF-KPCA implementation, which is implemented in the class ``RFF_KernelPCA``.



class KernelPCA(SklearnKernelPCA):
    """Kernel PCA implementation with an inverse transform method."""
    def __init__(self, n_components=None, kernel='rbf', gamma=None, degree=3, coef0=1, random_state=None):
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            random_state=random_state,
            fit_inverse_transform=False
        )
    
    def fit(self, X, y=None, fit_inverse_transform=False, alpha_bws=None, alpha_proposed=None):
        super().fit(X, y)
        if fit_inverse_transform:
            if alpha_bws is None:
                raise ValueError('Must provide `alpha_bws`')
            if alpha_proposed is None:
                raise ValueError('Must provide `alpha_proposed`')
            self.fit_inverse_transform_bws(alpha_bws)
            self.fit_inverse_transform_ours(alpha_proposed)
        return self
    
    def fit_inverse_transform_bws(self, alpha=None):
        """
            Precompute the necessary components for the inverse transform.
            This method is called during fitting to store the eigenvectors/eigenvalues of the Gram matrix for the pre-image mapping.
        """
        # get the projection of the original data in the kernel PCA space
        if not hasattr(self, '_preimage_eigenvals_'):
            X_transformed = self.transform(self.X_fit_)
            # compute the kernel matrix for the transformed data
            K = self._get_kernel(X_transformed)

            # compute the eigenvalues and eigenvectors of the kernel matrix
            eigenvals, eigenvectors = np.linalg.eigh(K)
            self._preimage_eigenvals_ = eigenvals
            self._preimage_eigenvectors_ = eigenvectors
            self.X_transformed_fit_ = X_transformed

        # calculate dual coefficients
        self.alpha_bws = alpha
        diag = 1 / (self._preimage_eigenvals_ + self.alpha_bws)
        self.dual_coeffs = self._preimage_eigenvectors_ @ (diag[:, None] *  (self._preimage_eigenvectors_.T @ self.X_fit_))
    
    def inverse_transform_bws(self, X_transformed, alpha=1e-12):
        ''''
            Modified sklearn's inverse_transform to include the ridge parameter at inference time.
            This is less expensive than refitting the model with a different alpha.
        '''
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling inverse_transform.")
       
        if not hasattr(self, 'dual_coeffs'):
            self.fit_inverse_transform_bws(alpha)
        
        K = self._get_kernel(X_transformed, self.X_transformed_fit_)
        return  np.dot(K, self.dual_coeffs)
        
    def inverse_transform_bws2(self, X_transformed, alpha=[1e-12]):
        ''''
            Efficient `inverse_transform_bws` for multiple alpha values.
        '''
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling inverse_transform.")
        if not hasattr(self, '_preimage_eigenvals_'):
            X_transformed_fit_ = self.transform(self.X_fit_)
            # compute the kernel matrix for the transformed data
            K = self._get_kernel(X_transformed_fit_)

            # compute the eigenvalues and eigenvectors of the kernel matrix
            eigenvals, eigenvectors = np.linalg.eigh(K)
            self._preimage_eigenvals_ = eigenvals
            self._preimage_eigenvectors_ = eigenvectors
            self.X_transformed_fit_ = X_transformed_fit_

        # compute the kernel matrix for the input data
        K = self._get_kernel(X_transformed, self.X_transformed_fit_)

        # precompute fixed elements
        U = K @ self._preimage_eigenvectors_
        V = self._preimage_eigenvectors_.T @ self.X_fit_
        
        # compute diagonal elements for each alpha (vectorized) 
        diag = 1 / (self._preimage_eigenvals_ + np.array(alpha).reshape(-1, 1))
        
        # compute the inverse transform for each alpha (vectorized)
        return  U @ (diag[:, :, None] * V)
    
    def fit_inverse_transform_ours(self, alpha):
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling inverse_transform.")

        # scale eigenvectors (properly account for null-space for dot product)
        non_zeros = np.flatnonzero(self.eigenvalues_)
        scaled_alphas = np.zeros_like(self.eigenvectors_)
        
        scaling_factor = np.sqrt(self.eigenvalues_[non_zeros]) / (
                self.eigenvalues_[non_zeros] + alpha * self.X_fit_.shape[0]
                )
        scaled_alphas[:, non_zeros] = self.eigenvectors_[:, non_zeros] * scaling_factor
            
        # compute centered original data
        self.alpha_ours = alpha
        self.weights = scaled_alphas.T @ (self.X_fit_ - self.X_fit_.mean(axis=0)) 

        # TODO: potential efficiency improvement by storing 
        # self.eigenvectors_[:, non_zeros].T @ (self.X_fit_ - self.X_fit_.mean(axis=0))
        # from call to inverse_transform_ours2

    def inverse_transform_ours(self, X_transformed):
        """
        Proposed inverse transform of the kernel PCA projection.

        Parameters:
        -----------
        X_transformed (numpy.ndarray): 
            The data in the kernel PCA space.

        
        Returns:
        --------
        X_original (numpy.ndarray): 
            The approximate original data reconstructed from the kernel PCA space.
        """    
        return X_transformed @ self.weights  + self.X_fit_.mean(axis=0)

    def inverse_transform_ours2(self, X_transformed, alpha=[0.0]):
        """
        Efficient `inverse_transform_proposed` for multiple alpha values.
        """
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling inverse_transform.")
        
        # scale eigenvectors, accounting for potential zero eigenvalues
        non_zeros = np.flatnonzero(self.eigenvalues_)
        
        scaling_factor = np.sqrt(self.eigenvalues_[non_zeros]) / (
                self.eigenvalues_[non_zeros] + np.array(alpha).reshape(-1, 1) * self.X_fit_.shape[0]
                )        
        scaled_alphas = self.eigenvectors_[:, non_zeros] * scaling_factor[:, None, :]
                    
        # compute centered original data
        X_fit_centered = self.X_fit_ - self.X_fit_.mean(axis=0) 
    
        # compute the inverse transform
        X_reconstructed = np.dot(X_transformed, np.moveaxis(scaled_alphas, 1, 2)) @ X_fit_centered + self.X_fit_.mean(axis=0)
        
        return np.moveaxis(X_reconstructed, 0, 1)  
        
# RFF-KPCA implementation for RBF kernel with the proposed inverse transform.
# Implements the random Fourier features approximation of the RBF kernel, which allows for a more scalable kernel PCA implementation.

class RFF_KernelPCA():
    """
    Random Fourier Features Kernel PCA implementation for RBF kernel.

    Pre-image methods:
    =================
    -- inverse_transform_bws: RFF version of the standard KPCA BWS method

    -- inverse_transform_ours: extension of our method to KPCA-RFF

    -- inverse_transform_tr: linear regression method of
        Tiddeman, Bernard, and Will Robinson. "Pre-image calculation for random Fourier feature kernel machines."
          UK Workshop on Computational Intelligence. Cham: Springer Nature Switzerland, 2023. 

    """
    def __init__(self, n_rffs, n_components=None, gamma=1.0, gamma_inv=None, random_state=None):
        '''
        Initialize the RFFKPCA object.
        
        Parameters:
        -----------
        - n_rffs (int): Number of random Fourier features.

        - n_components (int or None): Number of components to keep. If None, all components are kept.
        
        - gamma (float): Kernel coefficient for RBF kernel.
                    Note: we follow sklearn's implementation where gamma is the inverse of 
                        the standard deviation of the RBF kernel, i.e., gamma = 1/(2*sigma^2).
        
        - random_state (int or None): Random seed for reproducibility.
        '''
        if n_components == None:
            n_components = -1
        self.n_rffs = n_rffs
        self.n_components = n_components
        self.gamma = gamma
        self.gamma_inv = gamma if gamma_inv is None else gamma_inv
        self.random_state = random_state

        
    def fit(self, X):
        dim = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Sample random frequencies for RFF
        self.W = rng.normal(scale=np.sqrt(2 * self.gamma), size=(dim, self.n_rffs))
        self.b = rng.uniform(0, 2 * np.pi, size=self.n_rffs)
        self.rff_map = lambda x: np.sqrt(2) * np.cos(x @ self.W + self.b)

        # Compute the RFF features
        Z = self.rff_map(X) 
        
        # Center the RFF features
        Z_mean = Z.mean(axis=0)
        Z_centered = Z - Z_mean

        # Compute the covariance matrix and eigen decomposition
        cov_matrix = np.cov(Z_centered, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx][:self.n_components]
        self.eigenvectors_ = eigenvectors[:, idx][:, :self.n_components]

        # Store data for later use in inverse transform
        self.X_fit_ = X.copy()
        self.Z_fit_ = Z.copy()
        
    def transform(self, X):
        Z = np.sqrt(2) * np.cos(X @ self.W + self.b)
        
        non_zeros = np.flatnonzero(self.eigenvalues_)

        return (Z - self.Z_fit_.mean(axis=0)) @ self.eigenvectors_[:,non_zeros]

    def fit_inverse_transform_bws(self, alpha):
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling fit_inverse_transform.")

        if not hasattr(self, '_preimage_eigenvals_'):
            #===== generate RFFs ===============
            rng = np.random.RandomState(self.random_state)
            # Sample random frequencies for RFF
            n = self.n_components if self.n_components != -1 else len(self.eigenvalues_)
            self.preimage_W= rng.normal(scale=np.sqrt(2 * self.gamma_inv), size=(n, self.n_rffs))
            self.preimage_b = rng.uniform(0, 2 * np.pi, size=self.n_rffs)
            self.preimage_rff_map = lambda x: np.sqrt(2/self.n_rffs) * np.cos(x @ self.preimage_W + self.preimage_b) 

            #=========== evaluate RFF map at training data ============
            self.preimage_Z_fit = self.preimage_rff_map(self.transform(self.X_fit_))
            # compute the eigenvalues and eigenvectors 
            self._preimage_eigenvals_bws_, self._preimage_eigenvectors_bws_ = np.linalg.eigh(self.preimage_Z_fit.T @ self.preimage_Z_fit)
            # pre-compute fixed element
            self.fixed_M_bws = self._preimage_eigenvectors_bws_.T @ self.preimage_Z_fit.T @ self.X_fit_

        self.alpha_bws = alpha
        diag = 1 / (self._preimage_eigenvals_bws_ + self.alpha_bws)
        self.dual_coeffs = self._preimage_eigenvectors_bws_ @ (diag[:, None] * self.fixed_M_bws)
           
    def inverse_transform_bws(self, X_transformed, alpha=None):
        if not hasattr(self, 'dual_coeffs') or alpha is not None:
            self.fit_inverse_transform_bws(alpha)

        return self.preimage_rff_map(X_transformed) @ self.dual_coeffs

    def inverse_transform_bws2(self, X_transformed, alpha=[1e-12]):
        ''''
            Efficient `inverse_transform_bws` for multiple alpha values.
        '''
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling this function.")

        if not hasattr(self, '_preimage_eigenvals_bws_'):
            #===== generate RFFs ===============
            rng = np.random.RandomState(self.random_state)
            # Sample random frequencies for RFF
            n = self.n_components if self.n_components != -1 else len(self.eigenvalues_)
            self.preimage_W = rng.normal(scale=np.sqrt(2 * self.gamma_inv), size=(n, self.n_rffs))
            self.preimage_b = rng.uniform(0, 2 * np.pi, size=self.n_rffs)
            self.preimage_rff_map = lambda x: np.sqrt(2/self.n_rffs) * np.cos(x @ self.preimage_W + self.preimage_b) 

            #=========== evaluate RFF map at training data ============
            self.preimage_Z_fit = self.preimage_rff_map(self.transform(self.X_fit_))
            # compute the eigenvalues and eigenvectors 
            self._preimage_eigenvals_bws_, self._preimage_eigenvectors_bws_ = np.linalg.eigh(self.preimage_Z_fit.T @ self.preimage_Z_fit)
            # pre-compute fixed factor
            self.fixed_M_bws = self._preimage_eigenvectors_bws_.T @ self.preimage_Z_fit.T @ self.X_fit_

        # compute diagonal elements for each alpha (vectorized) 
        diag = 1 / (self._preimage_eigenvals_bws_ + np.array(alpha).reshape(-1, 1))        
        # compute the inverse transform for each alpha (vectorized)
        return  (self.preimage_rff_map(X_transformed) @ self._preimage_eigenvectors_bws_) @ (diag[:, :, None] * self.fixed_M_bws)

    def fit_inverse_transform_ours(self, alpha):
            if self.X_fit_ is None:
                raise ValueError("Must fit the model before calling fit_inverse_transform.")
            
            if not hasattr(self, 'fixed_M_ours'):
                non_zeros = np.flatnonzero(self.eigenvalues_)
                self.fixed_M_ours = (self.eigenvectors_[:, non_zeros].T @ (self.Z_fit_ - self.Z_fit_.mean(axis=0)).T) @ (self.X_fit_ - self.X_fit_.mean(axis=0))
            
            self.alpha = alpha
            n_samples = self.X_fit_.shape[0]
            # diagonal elements accounting for potential zero eigenvalues
            non_zeros = np.flatnonzero(self.eigenvalues_)
            diag = 1 / (n_samples * (self.eigenvalues_[non_zeros] + self.alpha))
            self.weights = diag[:, None] * self.fixed_M_ours

            #self.weights = ((self.Z_fit_ - self.Z_fit_.mean(axis=0)) @ (self.eigenvectors_[:, non_zeros] * diag)).T @ (self.X_fit_ - self.X_fit_.mean(axis=0))
    
    def inverse_transform_ours(self, X_transformed, alpha=None):
        if not hasattr(self, 'weights') or alpha is not None:
            self.fit_inverse_transform_ours(alpha)
        
        return self.X_fit_.mean(axis=0) + X_transformed @ self.weights
    
    def inverse_transform_ours2(self, X_transformed, alpha=[1e-12]):
        '''
        Evaluate the inverse transform with possibly multiple ridge paramater values for hyperparam tuning
        '''
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling this function.")
        
        if not hasattr(self, 'fixed_M_ours'):
            non_zeros = np.flatnonzero(self.eigenvalues_)
            self.fixed_M_ours = (self.eigenvectors_[:, non_zeros].T @ (self.Z_fit_ - self.Z_fit_.mean(axis=0)).T) @ (self.X_fit_ - self.X_fit_.mean(axis=0))
                    
        n_samples = self.X_fit_.shape[0]
        # diagonal elements accounting for potential zero eigenvalues
        non_zeros = np.flatnonzero(self.eigenvalues_)
        diag = 1 / (n_samples * (self.eigenvalues_[non_zeros] + np.array(alpha).reshape(-1, 1)))

        return self.X_fit_.mean(axis=0) + (X_transformed * diag[:, None, :]) @ self.fixed_M_ours

    def fit_inverse_transform_tr(self, alpha=0.0):
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling fit_inverse_transform.")

        if not hasattr(self, '_preimage_eigenvals_tr_'):
            # transformed data
            X_transformed = self.transform(self.X_fit_)
            # append 1 for the constant term
            X_transformed = np.column_stack([np.ones((X_transformed.shape[0], 1)), X_transformed])
            # compute the eigenvalues and eigenvectors 
            self._preimage_eigenvals_tr_, self._preimage_eigenvectors_tr_ = np.linalg.eigh(X_transformed.T @ X_transformed)
            # pre-compute fixed element
            self.fixed_M_tr = self._preimage_eigenvectors_tr_.T @ X_transformed.T @ self.X_fit_

        self.alpha_tr = alpha
        diag = 1 / (self._preimage_eigenvals_tr_ + self.alpha_tr)
        self.tr_weights = self._preimage_eigenvectors_tr_ @ (diag[:, None] * self.fixed_M_tr)

    def inverse_transform_tr(self, X_transformed, alpha=None):
        if not hasattr(self, 'tr_weights'):
            self.fit_inverse_transform_tr(alpha) if alpha else self.fit_inverse_transform_tr()
        # append 1 for the constant term
        X_transformed = np.column_stack([np.ones((X_transformed.shape[0], 1)), X_transformed])

        return X_transformed @ self.tr_weights

    # def inverse_transform_tr2(self, X_transformed, alpha=[1e-12]):
    #     ''''
    #         Efficient `inverse_transform_tr` for multiple alpha values.
    #     '''
    #     if self.X_fit_ is None:
    #         raise ValueError("Must fit the model before calling this method.")

    #     if not hasattr(self, '_preimage_eigenvals_tr_'):
    #         # transformed data
    #         X_transformed = self.transform(self.X_fit_)
    #         # append 1 for the constant term
    #         X_transformed = np.column_stack([np.ones((X_transformed.shape[0], 1)), X_transformed])
    #         # compute the eigenvalues and eigenvectors 
    #         self._preimage_eigenvals_tr_, self._preimage_eigenvectors_tr_ = np.linalg.eigh(X_transformed.T @ X_transformed)
    #         # pre-compute fixed element
    #         self.fixed_M_tr = self._preimage_eigenvectors_tr_.T @ X_transformed.T @ self.X_fit_

    #     # append 1 for the constant term
    #     X_transformed = np.column_stack([np.ones((X_transformed.shape[0], 1)), X_transformed])

    #     # compute diagonal elements for each alpha (vectorized) 
    #     diag = 1 / (self._preimage_eigenvals_tr_ + np.array(alpha).reshape(-1, 1))        
    #     # compute the inverse transform for each alpha (vectorized)
    #     print((X_transformed @ self._preimage_eigenvectors_tr_).shape)
    #     return  (X_transformed @ self._preimage_eigenvectors_tr_) @ (diag[:, :, None] * self.fixed_M_tr)

    def inverse_transform_tr2(self, X_transformed, alpha=[1e-12]):
        '''
        Efficient `inverse_transform_tr` for multiple alpha values.
        X_transformed: KPCA-RFF embedding of the points to invert, shape (m, q).
        Returns an array of shape (n_alpha, m, d).
        '''
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling this method.")

        # one-time setup, cached across calls
        if not hasattr(self, '_preimage_eigenvals_tr_'):
            Z_fit = self.transform(self.X_fit_)
            Z_fit = np.column_stack([np.ones((Z_fit.shape[0], 1)), Z_fit])  # constant term
            eigvals, eigvecs = np.linalg.eigh(Z_fit.T @ Z_fit)
            self._preimage_eigenvals_tr_ = np.maximum(eigvals, 0.0)  
            self._preimage_eigenvectors_tr_ = eigvecs
            self.fixed_M_tr = eigvecs.T @ Z_fit.T @ self.X_fit_

        # augment the points to invert with the constant term
        Z = np.column_stack([np.ones((X_transformed.shape[0], 1)), X_transformed])

        # diagonal ridge factors for each alpha: (n_alpha, q+1)
        diag = 1 / (self._preimage_eigenvals_tr_ + np.array(alpha).reshape(-1, 1))

        # (m, q+1) @ [(n_alpha, q+1, 1) * (q+1, d)] -> (n_alpha, m, d)
        return (Z @ self._preimage_eigenvectors_tr_) @ (diag[:, :, None] * self.fixed_M_tr)
    
    

