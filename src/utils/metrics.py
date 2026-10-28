import numpy as np
from skimage.metrics import structural_similarity as ssim

def batch_ssim(X, Y, data_range, rgb=False):
    # X and Y: arrays of shape (N, H, W)
    scores = []
    for x, y in zip(X, Y):
        score = ssim(x, y, data_range=data_range, channel_axis=-1) if rgb else ssim(x, y, data_range=data_range) 
        scores.append(score)
    return np.mean(scores)

def metrics(X, Y):
    ''' 
    Calculate MSE, SSIM, and PSNR for multiple images at once.
    '''
    data_range = X.max() - X.min()

    # X, Y: shape (N, H, W) or (N, H, W, C) 
    rgb = (X.ndim == 4)
    ssim_ = batch_ssim(X, Y, data_range=data_range, rgb=rgb)
    mse_ = np.mean((X - Y) ** 2, axis=tuple(range(1, X.ndim)))
    psnr_ = 10 * np.log10((data_range ** 2) / mse_)

    mse_, psnr_ = np.mean(mse_), np.mean(psnr_)

    return mse_, ssim_, psnr_

def multi_test_metrics(X, Y):
    # X.shape: (N, H, W) or (N, H, W, C)
    # Y.shape: (M, N, H, W) or (M, N, H, W, C)
    scores = np.zeros((Y.shape[0], 3))
    for i in range(Y.shape[0]):
        scores[i, :] = metrics(X, Y[i,...])
    
    return scores[:,0], scores[:,1], scores[:,2]