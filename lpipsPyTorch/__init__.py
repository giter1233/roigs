import torch

# Fallback-safe import: handle both module layout and direct file path
try:
    from .modules.lpips import LPIPS  # expected path
except ModuleNotFoundError:
    # Some environments may not index the subpackage properly
    # Attempt relative import by loading via importlib
    import importlib
    LPIPS = importlib.import_module('lpipsPyTorch.modules.lpips').LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
