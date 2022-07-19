import torch
import numpy as np
from skimage import metrics


def MSE(image1, image2, batch: bool = False):
    assert image1.shape == image2.shape
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()
    assert isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray)

    if batch:
        mse = sum(metrics.mean_squared_error(im1, im2) for im1, im2 in zip(image1, image2)) / image1.shape[0]
    else:
        mse = metrics.mean_squared_error(image1, image2)
    return mse


def PSNR(image1, image2, batch: bool = False):
    assert image1.shape == image2.shape
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()
    assert isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray)

    if batch:
        psnr = sum(metrics.peak_signal_noise_ratio(im1, im2) for im1, im2 in zip(image1, image2)) / image1.shape[0]
    else:
        psnr = metrics.peak_signal_noise_ratio(image1, image2)
    return psnr


def SSIM(image1, image2, batch: bool = False):
    assert image1.shape == image2.shape
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()
    assert isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray)

    if batch:
        ssim = sum(metrics.structural_similarity(im1, im2, channel_axis=0) for im1, im2 in zip(image1, image2)) / image1.shape[0]
    else:
        ssim = metrics.structural_similarity(image1, image2, channel_axis=0)
    return ssim
