#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

C1 = 0.01 ** 2
C2 = 0.03 ** 2

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ---------------------- High-frequency utilities ----------------------

def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """Ensure input is BxCxHxW float tensor."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    return x

def sobel_grad(img: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude (per-pixel), averaged over channels.
    Args:
        img: Tensor of shape (C,H,W) or (B,C,H,W) in [0,1].
    Returns:
        grad_mag: Tensor of shape (B,1,H,W)
    """
    x = _ensure_4d(img)
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=device, dtype=dtype)
    ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=device, dtype=dtype)
    kx = kx.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    grad = grad.mean(dim=1, keepdim=True)
    return grad

def laplacian(img: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian response magnitude, averaged over channels.
    Args:
        img: (C,H,W) or (B,C,H,W)
    Returns:
        lap_mag: (B,1,H,W)
    """
    x = _ensure_4d(img)
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device, dtype=dtype)
    k = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    l = F.conv2d(x, k, padding=1, groups=C)
    l = torch.abs(l).mean(dim=1, keepdim=True)
    return l

def fft_highband_loss(pred: torch.Tensor, gt: torch.Tensor, cutoff: float = 0.25, reduction: str = 'mean') -> torch.Tensor:
    """L1 loss on high-frequency band of the amplitude spectra.
    Args:
        pred, gt: (C,H,W) or (B,C,H,W) images in [0,1].
        cutoff: radial frequency threshold in [0, 0.5]. Larger -> higher frequencies.
        reduction: 'mean' or 'sum'.
    Returns:
        scalar loss tensor
    """
    p = _ensure_4d(pred)
    t = _ensure_4d(gt)
    assert p.shape == t.shape, "pred and gt must have the same shape"
    B, C, H, W = p.shape
    device = p.device

    # Frequency grid
    fy = torch.fft.fftfreq(H, device=device).view(H, 1).expand(H, W)
    fx = torch.fft.fftfreq(W, device=device).view(1, W).expand(H, W)
    radius = torch.sqrt(fx * fx + fy * fy)
    mask = (radius >= cutoff).float()
    mask = torch.fft.fftshift(mask)
    mask = mask.view(1, 1, H, W)

    # Amplitude spectra
    P = torch.fft.fftshift(torch.fft.fft2(p, norm='ortho'), dim=(-2, -1))
    T = torch.fft.fftshift(torch.fft.fft2(t, norm='ortho'), dim=(-2, -1))
    AmpP = torch.abs(P)
    AmpT = torch.abs(T)

    diff = torch.abs(AmpP - AmpT) * mask
    if reduction == 'mean':
        return diff.mean()
    elif reduction == 'sum':
        return diff.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def multiscale_ssim_loss(pred: torch.Tensor, gt: torch.Tensor, scales: list = [1, 2, 4], weights: list = [0.5, 0.3, 0.2]) -> torch.Tensor:
    """Multi-scale SSIM loss for better detail preservation.
    Args:
        pred, gt: (C,H,W) or (B,C,H,W) images in [0,1].
        scales: List of downsampling scales.
        weights: Weights for each scale.
    Returns:
        Combined multi-scale SSIM loss.
    """
    pred = _ensure_4d(pred)
    gt = _ensure_4d(gt)
    
    total_loss = 0.0
    for scale, weight in zip(scales, weights):
        if scale == 1:
            # Original resolution
            ssim_val = ssim(pred, gt)
        else:
            # Downsampled resolution
            pred_down = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            gt_down = F.avg_pool2d(gt, kernel_size=scale, stride=scale)
            ssim_val = ssim(pred_down, gt_down)
        
        total_loss += weight * (1.0 - ssim_val)
    
    return total_loss


def detail_aware_loss(pred: torch.Tensor, gt: torch.Tensor, 
                     highfreq_weight: float = 0.1, 
                     laplacian_weight: float = 0.05,
                     gradient_weight: float = 0.03) -> torch.Tensor:
    """Combined detail-aware loss for enhanced fine structure preservation.
    Args:
        pred, gt: (C,H,W) or (B,C,H,W) images in [0,1].
        highfreq_weight: Weight for high-frequency loss.
        laplacian_weight: Weight for Laplacian loss.
        gradient_weight: Weight for gradient loss.
    Returns:
        Combined detail loss.
    """
    # High-frequency loss
    hf_loss = fft_highband_loss(pred, gt, cutoff=0.25)
    
    # Laplacian loss for edge preservation
    pred_lap = laplacian(pred)
    gt_lap = laplacian(gt)
    lap_loss = F.l1_loss(pred_lap, gt_lap)
    
    # Gradient loss for texture preservation
    pred_grad = sobel_grad(pred)
    gt_grad = sobel_grad(gt)
    grad_loss = F.l1_loss(pred_grad, gt_grad)
    
    return highfreq_weight * hf_loss + laplacian_weight * lap_loss + gradient_weight * grad_loss


def enhanced_loss_function(pred: torch.Tensor, gt: torch.Tensor,
                          l1_weight: float = 0.8,
                          ssim_weight: float = 0.2,
                          multiscale_ssim_weight: float = 0.1,
                          detail_weight: float = 0.1) -> torch.Tensor:
    """Enhanced loss function combining multiple loss terms for better quality.
    Args:
        pred, gt: Predicted and ground truth images.
        l1_weight: Weight for L1 loss.
        ssim_weight: Weight for standard SSIM loss.
        multiscale_ssim_weight: Weight for multi-scale SSIM loss.
        detail_weight: Weight for detail-aware loss.
    Returns:
        Combined enhanced loss.
    """
    # Basic losses
    l1 = l1_loss(pred, gt)
    ssim_val = ssim(pred.unsqueeze(0) if pred.dim() == 3 else pred, 
                   gt.unsqueeze(0) if gt.dim() == 3 else gt)
    
    # Enhanced losses
    ms_ssim = multiscale_ssim_loss(pred, gt)
    detail_loss = detail_aware_loss(pred, gt)
    
    total_loss = (l1_weight * l1 + 
                  ssim_weight * (1.0 - ssim_val) +
                  multiscale_ssim_weight * ms_ssim +
                  detail_weight * detail_loss)
    
    return total_loss


# ---------------------- Dynamic Laplacian Pyramid Loss (LPDR) ----------------------

def _gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur with given kernel size and sigma using depthwise conv.
    Args:
        img: (C,H,W) or (B,C,H,W)
    Returns:
        Blurred image with same shape as input.
    """
    x = _ensure_4d(img)
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # Build 2D Gaussian kernel via existing gaussian() helper
    g1d = gaussian(kernel_size, sigma).to(device=device, dtype=dtype).unsqueeze(1)
    g2d = (g1d @ g1d.t()).unsqueeze(0).unsqueeze(0)  # (1,1,ks,ks)
    g2d = g2d / g2d.sum()
    kernel = g2d.expand(C, 1, kernel_size, kernel_size).contiguous()

    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=C)


def _build_laplacian_pyramid(img: torch.Tensor, levels: int = 3, kernel_size: int = 5, sigma: float = 1.0):
    """Build Laplacian pyramid levels for an image.
    Returns a list [L0, L1, ..., L_{levels-1}] and the final lowpass image (DC).
    """
    x = _ensure_4d(img)
    pyr = []
    current = x
    for _ in range(levels):
        blurred = _gaussian_blur(current, kernel_size=kernel_size, sigma=sigma)
        low = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        up = F.interpolate(low, size=current.shape[-2:], mode="bilinear", align_corners=False)
        lap = current - up
        pyr.append(lap)
        current = low
    return pyr, current


def laplacian_pyramid_dynamic_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    levels: int = 3,
    kernel_size: int = 5,
    sigma: float = 1.0,
    weight_gamma: float = 1.0,
    include_dc: bool = True,
    dc_weight: float = 0.1,
) -> torch.Tensor:
    """Dynamic Laplacian Pyramid loss (LPDR).
    Emphasizes frequencies proportionally to the GT energy at each pyramid level.

    Args:
        pred, gt: (C,H,W) or (B,C,H,W) images in [0,1]. Shapes must match.
        levels: Number of Laplacian pyramid levels.
        kernel_size: Gaussian kernel size for pyramid.
        sigma: Gaussian sigma for pyramid.
        weight_gamma: Nonlinear shaping for dynamic weights (>1 favors energetic bands).
        include_dc: Whether to include the low-frequency (DC) residual constraint.
        dc_weight: Weight for DC term relative to band losses.
    Returns:
        Scalar loss tensor.
    """
    p = _ensure_4d(pred)
    t = _ensure_4d(gt)
    assert p.shape == t.shape, "pred and gt must have the same shape"

    # Build pyramids
    p_pyr, p_dc = _build_laplacian_pyramid(p, levels=levels, kernel_size=kernel_size, sigma=sigma)
    t_pyr, t_dc = _build_laplacian_pyramid(t, levels=levels, kernel_size=kernel_size, sigma=sigma)

    # Compute GT energy per level and derive normalized dynamic weights
    energies = []
    for lvl in range(levels):
        # L1 energy proxy, averaged spatially and across channels
        e = t_pyr[lvl].abs().mean()
        energies.append(e)
    energies = torch.stack(energies)  # (levels,)
    energies = (energies + 1e-8)  # stability

    # Nonlinear shaping to emphasize more textured bands
    if weight_gamma != 1.0:
        energies = energies.pow(weight_gamma)

    weights = energies / energies.sum()

    # Weighted band loss (L1 between Laplacian bands)
    band_loss = 0.0
    for lvl in range(levels):
        band_loss = band_loss + weights[lvl] * F.l1_loss(p_pyr[lvl], t_pyr[lvl])

    # Optional DC constraint to stabilize color/brightness consistency
    if include_dc:
        dc_loss = F.l1_loss(p_dc, t_dc)
        total = band_loss + dc_weight * dc_loss
    else:
        total = band_loss

    return total

