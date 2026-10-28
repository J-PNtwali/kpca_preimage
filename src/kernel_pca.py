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
    """Random Fourier Features Kernel PCA implementation for RBF kernel."""
    def __init__(self, n_rffs, n_components=None, gamma=1.0, append_x=False, beta=1.0,  random_state=None):
        '''Initialize the RFFKPCA object.
        
        Parameters:
        -----------
        - n_rffs (int): Number of random Fourier features.
        - n_components (int or None): Number of components to keep. If None, all components are kept.
        - gamma (float): Kernel coefficient for RBF kernel.
                    Note: we follow sklearn's implementation where gamma is the inverse of 
                        the standard deviation of the RBF kernel, i.e., gamma = 1/(2*sigma^2).
        - append_x (bool, default: False): Whether or not to append the state vector into the RFF feature map
        - beta (float, default: 1.0): positive float. Scaling factor for the state vector. Ignored when `append_x = False`
        - random_state (int or None): Random seed for reproducibility.
        '''
        if n_components == None:
            n_components = -1
        self.n_rffs = n_rffs
        self.n_components = n_components
        self.gamma = gamma
        self.append_x = append_x
        self.beta = beta
        self.random_state = random_state

        
    def fit(self, X):
        n_samples, dim = X.shape
        rng = np.random.RandomState(self.random_state)

        # Sample random frequencies for RFF
        self.W = rng.normal(scale=np.sqrt(2 * self.gamma), size=(dim, self.n_rffs))
        self.b = rng.uniform(0, 2 * np.pi, size=self.n_rffs)

        # Compute the RFF features
        Z = np.sqrt(2) * np.cos(X @ self.W + self.b)
        if self.append_x:
            Z = np.column_stack((self.beta * X, Z))
            
            if self.n_components > (self.n_rffs + X.shape[-1]):
                self.n_components = -1

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

        return self
        
    def transform(self, X):
        Z = np.sqrt(2) * np.cos(X @ self.W + self.b)
        if self.append_x:
            Z = np.column_stack((self.beta * X, Z))
        
        non_zeros = np.flatnonzero(self.eigenvalues_)

        return (Z - self.Z_fit_.mean(axis=0)) @ self.eigenvectors_[:,non_zeros]

    def fit_inverse_transform(self, alpha):
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling fit_inverse_transform.")
        
        self.alpha = alpha
        n_samples = self.X_fit_.shape[0]
        # diagonal elements accounting for potential zero eigenvalues
        non_zeros = np.flatnonzero(self.eigenvalues_)
        diag = 1 / (n_samples * (self.eigenvalues_[non_zeros] + self.alpha))

        self.weights = ((self.Z_fit_ - self.Z_fit_.mean(axis=0)) @ (self.eigenvectors_[:, non_zeros]  * diag)).T @ (self.X_fit_ - self.X_fit_.mean(axis=0))

    def inverse_transform(self, X_transformed, alpha=1e-12):
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling inverse_transform.")
        
        return self.X_fit_.mean(axis=0) + X_transformed @ self.weights
    
    def inverse_transform2(self, X_transformed, alpha=[1e-12]):
        if self.X_fit_ is None:
            raise ValueError("Must fit the model before calling inverse_transform.")
        
        n_samples = self.X_fit_.shape[0]
        # diagonal elements accounting for potential zero eigenvalues
        non_zeros = np.flatnonzero(self.eigenvalues_)
        diag = 1 / (n_samples * (self.eigenvalues_[non_zeros] + np.array(alpha).reshape(-1, 1)))

        U = (self.eigenvectors_[:, non_zeros].T @ (self.Z_fit_ - self.Z_fit_.mean(axis=0)).T) @ (self.X_fit_ - self.X_fit_.mean(axis=0))

        return self.X_fit_.mean(axis=0) + (X_transformed * diag[:, None, :]) @ U
    

