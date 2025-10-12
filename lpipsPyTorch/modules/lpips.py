import torch
import torch.nn as nn
from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        super(LPIPS, self).__init__()
        self.net = get_network(net_type)
        self.lin = LinLayers(self.net.n_channels_list)

        # load pre-trained weights
        self.lin.load_state_dict(get_state_dict(net_type, version), strict=False)
        self.net.eval()
        self.lin.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # enforce 4D BCHW
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if y.dim() == 3:
            y = y.unsqueeze(0)

        # ensure RGB range roughly in [0,1]; refrain from clamping to allow gradients
        x_feats = self.net(x)
        y_feats = self.net(y)

        diffs = []
        for xf, yf, lin in zip(x_feats, y_feats, self.lin):
            diffs.append(lin((xf - yf) ** 2))

        # average spatially then sum over layers
        vals = [d.mean([2, 3]).squeeze(1) for d in diffs]
        val = torch.sum(torch.stack(vals, dim=0), dim=0)
        # return scalar if batch==1, else mean over batch
        return val.mean()