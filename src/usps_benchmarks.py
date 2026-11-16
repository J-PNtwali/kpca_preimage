import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from src.kernel_pca import KernelPCA, RFF_KernelPCA
from src.preimage_algorithms import mika_etal_preimage, kwok_tsang_preimage, honeine_rich_preimage
from src.utils import metrics
from typing import Union, List

import time
import tracemalloc
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class TimingResult:
    elapsed: float = 0.0
    peak_kb: float = 0.0

@contextmanager
def timer_and_memory(result: TimingResult):
    """Context manager that writes wall-clock time and peak memory into result."""
    tracemalloc.start()
    t0 = time.perf_counter()
    yield
    result.elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.peak_kb = peak / 1024


def get_usps_data(n_train=2_000, n_test=1_000, n_val=200, noise_scale=0.1, noise_train=False, rescale=False, seed:int=0):
    #==== Load the USPS digits dataset
    X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
    if rescale:
        X = MinMaxScaler().fit_transform(X)

    #==== train_val, test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify=y, random_state=seed, train_size=(n_train + n_val), test_size=n_test
        ) 
     #==== split `train_val` into `train` and `val`. The `val` dataset is used to optimize the ridge parameter
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, stratify=y_train_val, random_state=seed, train_size=n_train, test_size=n_val
        )

    #==== corrupt test and validation data with Gaussian noise 
    if noise_scale > 0.0:
        rng = np.random.RandomState(seed)
        noise = rng.normal(scale=noise_scale, size=X_test.shape)
        X_test_noisy = X_test + noise

        noise = rng.normal(scale=noise_scale, size=X_val.shape)
        X_val_noisy = X_val + noise

        if noise_train:
            noise = rng.normal(scale=noise_scale, size=X_train.shape)
            X_train += noise
    else:
        X_test_noisy = X_test.copy()
        X_val_noisy = X_val.copy()

    data = {'X_train': X_train, 'X_test': X_test, 'X_test_noisy': X_test_noisy, 'X_val':X_val, 'X_val_noisy':X_val_noisy}
    targets = {'y_train': y_train, 'y_test': y_test, 'y_val':y_val}

    return data, targets

def ridge_optimizer(kpca_obj:object, X_val:np.array, X_val_noisy:np.array=None,
                     baseline=True, method=None, return_best=True, lower=1e-12, upper=1e+1, n_alpha=10):
    if X_val_noisy is None:
        X_val_noisy = X_val

    # obtain reconstruction for different values of alpha
    alphas = np.geomspace(lower, upper, n_alpha)
    if isinstance(kpca_obj, KernelPCA):
        if baseline:
            Xs = kpca_obj.inverse_transform_bws2(kpca_obj.transform(X_val_noisy), alpha=alphas)
        else:
            Xs = kpca_obj.inverse_transform_ours2(kpca_obj.transform(X_val_noisy), alpha=alphas)
    
    if isinstance(kpca_obj, RFF_KernelPCA):
        assert method is not None, "method should not be None"
        if method == 'ours':
            Xs = kpca_obj.inverse_transform_ours2(kpca_obj.transform(X_val_noisy), alpha=alphas)
        elif method == 'bws':
            Xs = kpca_obj.inverse_transform_bws2(kpca_obj.transform(X_val_noisy), alpha=alphas)
        elif method == 'tr':
            Xs = kpca_obj.inverse_transform_tr2(kpca_obj.transform(X_val_noisy), alpha=alphas)
        else:
            raise ValueError("Invalid method. Must be either of {'ours', 'bws','tr'}.")
    
    # calculate metrics for each of the reconstructions
    mse_, ssim_, psnr_ = metrics.multi_test_metrics(X_val.reshape(-1, 16, 16), Xs.reshape(n_alpha, -1, 16, 16))

    if return_best:
        # get the best ``alpha`` based on MSE
        id = np.argmin(mse_)
        out = {'alpha':alphas[id], 'mse':mse_[id], 'ssim':ssim_[id], 'psnr':psnr_[id]}
    else:
        out = {'alpha':alphas, 'mse':mse_, 'ssim':ssim_, 'psnr':psnr_}
    
    return out

def run_benchmark(n_components:int, methods:Union[List[str], str]='all', kernel:str='rbf', gamma:float=0.0005, degree:int=3,
                n_train:int=2_000, n_test:int=1_000, n_val:int=200, noise_scale:float=0.1, noise_train:bool=False,
                rescale:bool=False, alpha_bakir:float=None, alpha_ours:float=None, n_alpha:int=10,
                ):
    '''Run the benchmark for USPS digits denoising using PCA, KPCA, and RFF-KPCA.

    Parameters
    ----------
    n_components : int
        Number of components for PCA and KPCA.
    methods : list or str, default='all'
        List of methods to include in the benchmark, default 'all'. Note that valid methods are
        ['bakir', 'mika', 'kwok', 'abrahamsen', 'honeine']
        where the method-name corresponds to last name of the first author of the pre-image algorithm.
        The proposed method is denoted by 'ours' and is always included in the benchmark.
    kernel : str, default='rbf'
        Kernel function for KPCA.
    gamma : float, default=0.0005
        Kernel coefficient for RBF kernel.
    degree : int, default=3
        Degree for polynomial kernel.
    n_train : int, default=2_000
        Number of training samples.
    n_test : int, default=1_000
        Number of test samples.
    n_val : int, default=200
        Number of validation samples.
    noise_scale : float, default=0.1
        Scale of the noise added to the data.
    noise_train : bool, default=False
        Whether to add noise to the training data.
    rescale : bool, default=False
        Whether to rescale the data.
    alpha_bakir : float, default=None
        Regularization parameter for the BWS method.
    alpha_ours : float, default=None
        Regularization parameter for the ours method.
    n_alpha : int, default=10
        Number of alpha values for hyperparameter tuning.
    '''
    # get train, val, test data
    data, _ = get_usps_data(n_train, n_test, n_val, noise_scale, noise_train, rescale)

    #==== Instatiate and fit kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, random_state=42)
    kpca.fit(data['X_train'])    

    if methods == 'all':
        methods = ['bakir', 'mika', 'kwok', 'abrahamsen', 'honeine']
    
    #==== optimize the ``alpha`` hyperparam based on validation data
    alpha_l, alpha_u = 1e-12, 1e-0
    if alpha_ours is None:
            if noise_scale > 0.0:  # use the regularized version for noisy test data only for nonlinear kernels
                alpha_ours = ridge_optimizer(kpca, data['X_val'], data['X_val_noisy'], baseline=False, lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']
            else:
                alpha_ours = 0.0
    kpca.fit_inverse_transform_ours(alpha=alpha_ours)
    
    if 'bakir' in methods:
        if alpha_bakir is None:
            alpha_bakir = ridge_optimizer(kpca, data['X_val'], data['X_val_noisy'], baseline=True, lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']
        kpca.fit_inverse_transform_bws(alpha=alpha_bakir)
    
    #==== reconstruction on testing data
    X_test_transformed = kpca.transform(data['X_test_noisy'])
    X_kpca_ours = kpca.inverse_transform_ours(X_test_transformed)
    
    if 'bakir' in methods:
        X_kpca_bakir = kpca.inverse_transform_bws(X_test_transformed)
    if 'kwok' in methods:  
        X_kpca_kwok = kwok_tsang_preimage(kpca, X_test_transformed, n=100)  
    if 'honeine' in methods:
        X_kpca_honeine = honeine_rich_preimage(kpca, X_test_transformed, lambda_=1e-9) 
    
    if 'mika' in methods or 'abrahamsen' in methods:
        X_kpca_mika = []
        X_kpca_abrahamsen = []
        
        for i in range(data['X_test_noisy'].shape[0]):
            x_transformed = X_test_transformed[i].copy()
            x0 = data['X_test_noisy'][i][np.newaxis,:]
            if 'mika' in methods:
                X_kpca_mika.append(mika_etal_preimage(kpca, x_transformed, x0, lambda_=0.0, max_iter=500, tol=1e-8)[0])
            if 'abrahamsen' in methods:
                X_kpca_abrahamsen.append(mika_etal_preimage(kpca, x_transformed, x0, x_ref=x0.copy(), lambda_=1e-4, max_iter=500, tol=1e-8)[0])
        if 'mika' in methods:
            X_kpca_mika = np.vstack(X_kpca_mika)  
        if 'abrahamsen' in methods:  
            X_kpca_abrahamsen = np.vstack(X_kpca_abrahamsen) 

    #==== calculate metrics
    out_data = {
        'test': data['X_test'], 
        'test_noisy': data['X_test_noisy'], 
        'ours': X_kpca_ours,
        'bakir': X_kpca_bakir if 'bakir' in methods else None, 
        'mika': X_kpca_mika if 'mika' in methods else None, 
        'abrahamsen': X_kpca_abrahamsen if 'abrahamsen' in methods else None, 
        'kwok': X_kpca_kwok if 'kwok' in methods else None, 
        'honeine': X_kpca_honeine if 'honeine' in methods else None}
    
    out_metrics = {key: metrics.metrics(data['X_test'].reshape(-1, 16, 16), value.reshape(-1, 16, 16)) for key, value in out_data.items() if value is not None and key != 'test'}
    
    return out_data, out_metrics

def run_efficiency_benchmark(num_components:Union[List[int], int]=50, num_train:Union[List[int], int]=2_000, num_test:Union[List[int], int]=1_000, 
                            seed:int=42, num_repeats:int=1, sigma:Union[List[float], float]=50, n_val:int=100, noise_scale:float=0.25, 
                            alpha_ours:float=1e-4, alpha_bakir:float=1e-4, n_alpha:int=10):
    '''Run the efficiency benchmark for USPS digits denoising using KPCA with RBF kernel.

    Parameters
    ----------
    num_components : int or list, default=50
        List of number of components for KPCA.
    num_train : int or list, default=2_000
        List of number of training samples.
    num_test : int or list, default=1_000
        List of number of test samples.
    seed : int, default=42
        Random seed for reproducibility.
    num_repeats : int, default=1
        Number of repeats for each configuration.
    sigma : float or list, default=50
        Bandwidth parameter for RBF kernel.
    n_val : int, default=100
        Number of validation samples.
    noise_scale : float, default=0.25
        Scale of the noise added to the data.
    alpha_ours : float, default=1e-4
        Regularization parameter for the proposed method. If None, it will be optimized using validation data.
    alpha_bakir : float, default=1e-4
        Regularization parameter for the BWS method. If None, it will be optimized using validation data.
    n_alpha : int, default=10
        Number of alpha values for hyperparameter tuning.
    '''
    if isinstance(num_components, int):
        num_components = [num_components]
    if isinstance(num_train, int):
        num_train = [num_train]
    if isinstance(num_test, int):
        num_test = [num_test]
    if not isinstance(sigma, list):
        sigma = [sigma]

    alpha_l, alpha_u = 1e-12, 1e-0

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10**9, size=num_repeats)

    
    all_metrics, all_timing = [], []
    for i, seed in enumerate(seeds):
        if i > 0:
                print("="*75)
        print(f"seed: {seed}")
        
        for j, n_train in enumerate(num_train):
            print("-"*50)
            print(f"n_train: {n_train}")

            # get train, val, test data
            data, _ = get_usps_data(n_train, max(num_test), n_val, noise_scale, noise_train=True, rescale=False, seed=seed)

            for k, n_components in enumerate(num_components):
                end = '\n' if k == len(num_components)-1 else ''
                print(" "*5 + f"n_components: {n_components:3}")

                for l, s in enumerate(sigma):
                    print(" "*10 + f"sigma: {s:3}")
                    
                    gamma = 1 / (2 * s**2)

                    timing_ours_fit, timing_bakir_fit = {}, {}
                    
                    #==== fitting KPCA
                    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma, random_state=seed)
                    kpca.fit(data['X_train']) 

                    # KPCA projection (all test data)
                    X_transformed_all = kpca.transform(data['X_test_noisy'])
                    
                    #===== fit inverse transform (including ridge parameter tuning) 
                    t = TimingResult()
                    with timer_and_memory(t):
                        if alpha_ours is None:
                            alpha_ours = ridge_optimizer(kpca, data['X_val'], data['X_val_noisy'], baseline=False, lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']                
                        kpca.fit_inverse_transform_ours(alpha=alpha_ours)
                    timing_ours_fit['fit_inverse'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                    #===== fit BWS inverse transform (including ridge parameter tuning) 
                    t = TimingResult()
                    with timer_and_memory(t):
                        if alpha_bakir is None:
                            alpha_bakir = ridge_optimizer(kpca, data['X_val'], data['X_val_noisy'], baseline=True, lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']                
                        kpca.fit_inverse_transform_bws(alpha=alpha_bakir)
                    timing_bakir_fit['fit_inverse'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                    #-------------------------------------------------------------------
                    #     KPCA projection and reconstruction for different test sizes
                    #-------------------------------------------------------------------
                    for m, n_test in enumerate(num_test):
                        end = '\n' if m == len(num_test)-1 else ' || '
                        print(" "*15 + f"n_test: {n_test:4}", end=end)

                        timing_ours = timing_ours_fit.copy()
                        timing_bakir = timing_bakir_fit.copy()
                        timing_honeine, timing_mika, timing_kwok = {}, {}, {}
                        
                        X_transformed = X_transformed_all[:n_test]

                        # KPCA reconstruction --- Ours
                        t = TimingResult()
                        with timer_and_memory(t):
                            X_reconstructed_ours = kpca.inverse_transform_ours(X_transformed) 
                        timing_ours[f'reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                        # KPCA reconstruction --- BWS (Bakir et al.)
                        t = TimingResult()
                        with timer_and_memory(t):
                            X_reconstructed_bakir = kpca.inverse_transform_bws(X_transformed) 
                        timing_bakir[f'reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                        # KPCA reconstruction --- Honeine & Richard (2011)
                        t = TimingResult()
                        with timer_and_memory(t):
                            X_reconstructed_honeine = honeine_rich_preimage(kpca, X_transformed, lambda_=1e-9)
                        timing_honeine[f'reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                        # KPCA reconstruction --- Mika et al. (1999)
                        t = TimingResult()
                        with timer_and_memory(t):
                            X_reconstructed_mika = []
                            for i in range(n_test):
                                x_transformed = X_transformed[i].copy()
                                x0 = data['X_test_noisy'][i][np.newaxis,:]
                                X_reconstructed_mika.append(mika_etal_preimage(kpca, x_transformed, x0, lambda_=0.0, max_iter=500, tol=1e-8)[0])
                            X_reconstructed_mika = np.vstack(X_reconstructed_mika)
                        timing_mika[f'reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                        # KPCA reconstruction --- Kwok & Tsang (2004)
                        t = TimingResult()
                        with timer_and_memory(t):
                            X_reconstructed_kwok = kwok_tsang_preimage(kpca, X_transformed, n=100)
                        timing_kwok[f'reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}                    

                        #-----------------------------------------------------------------
                        #            Metrics                                             #
                        #----------------------------------------------------------------- 
                        metrics_ours = metrics.metrics(data['X_test'][:n_test].reshape(-1, 16, 16), X_reconstructed_ours.reshape(-1, 16, 16))
                        metrics_bakir = metrics.metrics(data['X_test'][:n_test].reshape(-1, 16, 16), X_reconstructed_bakir.reshape(-1, 16, 16))
                        metrics_honeine = metrics.metrics(data['X_test'][:n_test].reshape(-1, 16, 16), X_reconstructed_honeine.reshape(-1, 16, 16))
                        metrics_mika = metrics.metrics(data['X_test'][:n_test].reshape(-1, 16, 16), X_reconstructed_mika.reshape(-1, 16, 16))
                        metrics_kwok = metrics.metrics(data['X_test'][:n_test].reshape(-1, 16, 16), X_reconstructed_kwok.reshape(-1, 16, 16))

                        out_metrics = {'ours': metrics_ours, 'bakir': metrics_bakir, 'honeine': metrics_honeine, 'mika': metrics_mika, 'kwok': metrics_kwok,
                                       'sigma': s, 'n_train':n_train, 'n_test':n_test, 'n_components': n_components, 'seed': seed}
                        all_metrics.append(pd.DataFrame(out_metrics))


                        #= merge timing
                        out_timing =  {'ours': timing_ours, 'bakir': timing_bakir, 'honeine': timing_honeine, 'mika': timing_mika, 'kwok': timing_kwok,
                                       'sigma': s, 'n_train':n_train, 'n_test':n_test, 'n_components': n_components, 'seed': i+1}
                        
                        all_timing.append(out_timing)
                
    return all_metrics, all_timing
    

def run_benchmark_rff(n_components:int, Sigma:list, num_rffs:list, num_samples:list, seed:int=42, n_test:int=1_000,
                    n_val:int=200, noise_scale:float=0.25, noise_train:bool=True, rescale:bool=False, n_alpha:int=20):
    
    all_metrics, all_timing = [], []

        
    for i, n_train in enumerate(num_samples):
        print("-"*50)
        print(f"n_train: {n_train}")
        
        # get train, val, test data
        data, _ = get_usps_data(n_train, n_test, n_val, noise_scale, noise_train, rescale, seed)

        for j, sigma in enumerate(Sigma):
            print(" "*5 + f"sigma: {sigma}")
            gamma = 1 / (2 * sigma**2)
            #-----------------------------------------------------------------
            #            Exact kernel PCA once                               #
            #-----------------------------------------------------------------
            kpca_timing = {}
            #==== fitting KPCA
            t = TimingResult()
            with timer_and_memory(t):
                kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma, random_state=seed)
                kpca.fit(data['X_train']) 

            kpca_timing['fit'] = {'time_s': t.elapsed,'peak_memory_kb': t.peak_kb}
            
            #===== fit inverse transform (including ridge parameter tuning) 
            alpha_l, alpha_u = 1e-12, 1e-0
            t = TimingResult()
            with timer_and_memory(t):
                alpha_ours = ridge_optimizer(kpca, data['X_val'], data['X_val_noisy'], baseline=False, lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']                
                kpca.fit_inverse_transform_ours(alpha=alpha_ours)
            kpca_timing['fit_inverse'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

            #===== KPCA projection
            t = TimingResult()
            with timer_and_memory(t):
                X_kpca_transformed = kpca.transform(data['X_test_noisy'])
            kpca_timing['projection'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

            #===== reconstruction
            t = TimingResult()
            with timer_and_memory(t):
                X_kpca_reconstructed = kpca.inverse_transform_ours(X_kpca_transformed) 
            kpca_timing['reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

            #===== calculate metrics 
            kpca_metrics = metrics.metrics(data['X_test'].reshape(-1, 16, 16), X_kpca_reconstructed.reshape(-1, 16, 16))

            #-----------------------------------------------------------------
            #            RFF-KPCA by number of n_rffs                        #
            #-----------------------------------------------------------------
            for k, n_rffs in enumerate(num_rffs):
                end = '\n' if k == len(num_rffs)-1 else ''  
                empty = 10 if k == 0 else 1
                print(" "*empty + f"n_rffs: {n_rffs} |", end=end)

                rff_timing = {}
                
                #==== fitting KPCA-RFF
                t = TimingResult()
                with timer_and_memory(t):
                    kpca_rff = RFF_KernelPCA(n_components=n_components, n_rffs=n_rffs, gamma=kpca.gamma, random_state=seed, append_x=False)
                    _ = kpca_rff.fit(data['X_train'])

                rff_timing['fit'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                    

                #===== fit inverse transform (including ridge parameter tuning) 
                t = TimingResult()
                with timer_and_memory(t):
                    alpha_rff = ridge_optimizer(kpca_rff, data['X_val'], data['X_val_noisy'], lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']
                    kpca_rff.fit_inverse_transform(alpha=alpha_rff)

                rff_timing['fit_inverse'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                    

                #===== RFF projection
                t = TimingResult()
                with timer_and_memory(t):
                    X_rff_transformed = kpca_rff.transform(data['X_test_noisy'])
                rff_timing['projection'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                    
                #===== reconstruction
                t = TimingResult()
                with timer_and_memory(t):
                    X_rff_reconstructed = kpca_rff.inverse_transform(X_rff_transformed) 
                rff_timing['reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                    

                #-----------------------------------------------------------------
                #            Metrics                                             #
                #----------------------------------------------------------------- 
                rff_metrics = metrics.metrics(data['X_test'].reshape(-1, 16, 16), X_rff_reconstructed.reshape(-1, 16, 16))
                
                out_metrics = {'kpca': kpca_metrics, 'rff': rff_metrics, 'sigma': sigma, 'n_train':n_train, 'n_rff': n_rffs, 'seed': seed}
                all_metrics.append(pd.DataFrame(out_metrics))

                #===== merge timing
                out_timing =  {'kpca': kpca_timing, 'rff': rff_timing, 'sigma': sigma, 'n_train':n_train, 'n_rff': n_rffs, 'seed': seed}
                all_timing.append(out_timing)
                
    return all_metrics, all_timing


def run_efficiency_benchmark_rff(n_components:int, Sigma:list, num_rffs:list, num_samples:list, seeds:list, n_test:int=1_000,
                    n_val:int=200, noise_scale:float=0.25, noise_train:bool=True, rescale:bool=False, n_alpha:int=20):
    
    all_metrics, all_timing = [], []

    for l, seed in enumerate(seeds):
        if l > 0:
                print("="*75)
        print(f"seed: {seed}")
        
        for i, n_train in enumerate(num_samples):
            print("-"*50)
            print(f"n_train: {n_train}")
            
            # get train, val, test data
            data, _ = get_usps_data(n_train, n_test, n_val, noise_scale, noise_train, rescale, seed)

            for j, sigma in enumerate(Sigma):
                print(" "*5 + f"sigma: {sigma}")
                gamma = 1 / (2 * sigma**2)
                #-----------------------------------------------------------------
                #            Exact kernel PCA once                               #
                #-----------------------------------------------------------------
                kpca_timing = {}
                #==== fitting KPCA
                t = TimingResult()
                with timer_and_memory(t):
                    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma, random_state=seed)
                    kpca.fit(data['X_train']) 

                kpca_timing['fit'] = {'time_s': t.elapsed,'peak_memory_kb': t.peak_kb}
                
                #===== fit inverse transform (including ridge parameter tuning) 
                alpha_l, alpha_u = 1e-12, 1e-0
                t = TimingResult()
                with timer_and_memory(t):
                    alpha_ours = ridge_optimizer(kpca, data['X_val'], data['X_val_noisy'], baseline=False, lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']                
                    kpca.fit_inverse_transform_ours(alpha=alpha_ours)
                kpca_timing['fit_inverse'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                #===== KPCA projection
                t = TimingResult()
                with timer_and_memory(t):
                    X_kpca_transformed = kpca.transform(data['X_test_noisy'])
                kpca_timing['projection'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                #===== reconstruction
                t = TimingResult()
                with timer_and_memory(t):
                    X_kpca_reconstructed = kpca.inverse_transform_ours(X_kpca_transformed) 
                kpca_timing['reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}

                #===== calculate metrics 
                kpca_metrics = metrics.metrics(data['X_test'].reshape(-1, 16, 16), X_kpca_reconstructed.reshape(-1, 16, 16))

                #-----------------------------------------------------------------
                #            RFF-KPCA by number of n_rffs                        #
                #-----------------------------------------------------------------
                for k, n_rffs in enumerate(num_rffs):
                    end = '\n' if k == len(num_rffs)-1 else ''  
                    empty = 10 if k == 0 else 1
                    print(" "*empty + f"n_rffs: {n_rffs} |", end=end)

                    rff_timing = {}
                    
                    #==== fitting KPCA-RFF
                    t = TimingResult()
                    with timer_and_memory(t):
                        kpca_rff = RFF_KernelPCA(n_components=n_components, n_rffs=n_rffs, gamma=kpca.gamma, random_state=seed, append_x=False)
                        _ = kpca_rff.fit(data['X_train'])

                    rff_timing['fit'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                        

                    #===== fit inverse transform (including ridge parameter tuning) 
                    t = TimingResult()
                    with timer_and_memory(t):
                        alpha_rff = ridge_optimizer(kpca_rff, data['X_val'], data['X_val_noisy'], lower=alpha_l, upper=alpha_u, n_alpha=n_alpha)['alpha']
                        kpca_rff.fit_inverse_transform(alpha=alpha_rff)

                    rff_timing['fit_inverse'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                        

                    #===== RFF projection
                    t = TimingResult()
                    with timer_and_memory(t):
                        X_rff_transformed = kpca_rff.transform(data['X_test_noisy'])
                    rff_timing['projection'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                        
                    #===== reconstruction
                    t = TimingResult()
                    with timer_and_memory(t):
                        X_rff_reconstructed = kpca_rff.inverse_transform(X_rff_transformed) 
                    rff_timing['reconstruction'] = {'time_s': t.elapsed, 'peak_memory_kb': t.peak_kb}
                        

                    #-----------------------------------------------------------------
                    #            Metrics                                             #
                    #----------------------------------------------------------------- 
                    rff_metrics = metrics.metrics(data['X_test'].reshape(-1, 16, 16), X_rff_reconstructed.reshape(-1, 16, 16))
                    
                    out_metrics = {'kpca': kpca_metrics, 'rff': rff_metrics, 'sigma': sigma, 'n_train':n_train, 'n_rff': n_rffs, 'seed': seed}
                    all_metrics.append(pd.DataFrame(out_metrics))

                    #===== merge timing
                    out_timing =  {'kpca': kpca_timing, 'rff': rff_timing, 'sigma': sigma, 'n_train':n_train, 'n_rff': n_rffs, 'seed': seed}
                    all_timing.append(out_timing)
                
    return all_metrics, all_timing