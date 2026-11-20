import numpy as np
import pandas as pd
from src.kernel_pca import KernelPCA, RFF_KernelPCA
from src.utils import metrics

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


def ridge_optimizer(kpca_obj:object, X_val:np.array, X_val_noisy:np.array=None,
                     baseline=True, method=None, return_best=True, lower=1e-12, upper=1e+1, n_alpha=10, width=16, height=16):
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
    mse_, ssim_, psnr_ = metrics.multi_test_metrics(X_val.reshape(-1, width, height), Xs.reshape(n_alpha, -1, width, height))

    if return_best:
        # get the best ``alpha`` based on MSE
        id = np.argmin(mse_)
        out = {'alpha':alphas[id], 'mse':mse_[id], 'ssim':ssim_[id], 'psnr':psnr_[id]}
    else:
        out = {'alpha':alphas, 'mse':mse_, 'ssim':ssim_, 'psnr':psnr_}
    
    return out

def timing_to_df_rff(out_timing) -> pd.DataFrame:
    """
    Flatten a single run's timing dict into a tidy dataframe row.
    params: the hyperparameters for this run (n_components, n_rffs, etc.)
    """
    methods = ['ours', 'bws', 'tr']
    params = {k: v for k, v in out_timing.items() if k not in methods}

    all_dfs = [pd.DataFrame(out_timing[method]).assign(method=method) for method in methods]
    out_timing_df = pd.concat(all_dfs, axis=0).assign(**params)
    return out_timing_df