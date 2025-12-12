# file: p4m_cnn.py
# A self-contained p4m (D4) equivariant CNN stack for classification
# Author: you + ChatGPT
# Notes:
#   - Uses exact 90° rotations (rot90) and reflections (flip) → lattice-exact equivariance
#   - Group axis order: 0..3 = rotations r^k (k=0,1,2,3), 4..7 = reflections m r^k
#   - All code is CPU/GPU agnostic; autograd works end-to-end.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.equivariant_modules.utils as utils

from models.equivariant_modules.p4mConv2d import P4M_LiftConv2d, P4M_GroupConv2d_Fast, P4M_BatchNorm, P4M_ReLU
from models.equivariant_modules.p4mMaxPooling2d import P4M_MaxPool2d, P4M_OrientationPool
# ---------------------------
# Building blocks and full model
# ---------------------------

class P4M_Block(nn.Module):
    """
    A convenient block: GroupConv -> BN -> ReLU -> (optional equivariant spatial pooling)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_ks=None, pool_pad=None):
        super().__init__()
        self.gconv = P4M_GroupConv2d_Fast(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn    = P4M_BatchNorm(out_channels)
        self.relu  = P4M_ReLU(inplace=True)
        if pool_ks is not None:
            if pool_pad is None:
                pool_pad = (pool_ks - 1)//2 if isinstance(pool_ks, int) else tuple((k-1)//2 for k in pool_ks)
            self.pool = P4M_MaxPool2d(kernel_size=pool_ks, padding=pool_pad)
        else:
            self.pool = None

    def forward(self, x):
        x = self.gconv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class P4M_EquivariantCNN(nn.Module):
    """
    End-to-end p4m-equivariant classifier.
    Pipeline:
        LiftConv -> [ P4M_Block x L ] -> OrientationPool (invariant) -> GAP -> Linear
    By default uses stride-free spatial pooling; adjust 'pool_ks' if you want mild smoothing.
    """
    def __init__(
        self,
        in_channels=1,          # e.g., MNIST=1, RGB=3
        channels=(16, 32, 64),  # feature widths after lift and blocks
        num_classes=10,
        kernel_size=3,
        pool_ks=3,              # equivariant spatial pooling kernel (None to disable)
        orient_pool='avg'       # 'avg' | 'max' | 'l2'
    ):
        super().__init__()
        k, p = kernel_size, (kernel_size-1)//2

        # Lift Z^2 -> p4m
        self.lift = P4M_LiftConv2d(in_channels, channels[0], kernel_size=k, padding=p)
        self.lift_bn = P4M_BatchNorm(channels[0])
        self.lift_relu = P4M_ReLU(inplace=True)
        self.lift_pool = P4M_MaxPool2d(kernel_size=pool_ks, padding=(pool_ks-1)//2) if pool_ks else None

        # Stacked p4m blocks
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks.append(P4M_Block(c_in, c_out, kernel_size=k, padding=p,
                                    pool_ks=pool_ks, pool_pad=(pool_ks-1)//2 if pool_ks else None))
        self.blocks = nn.Sequential(*blocks)

        # Orientation pooling -> invariant feature map
        self.gpool = P4M_OrientationPool(mode=orient_pool)

        # Classifier head: global average pooling over H,W then Linear
        self.head_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head_fc  = nn.Linear(channels[-1], num_classes)

        # Init head
        nn.init.kaiming_normal_(self.head_fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.head_fc.bias)

    def forward_features(self, x):
        """
        Returns: equivariant p4m maps after the block stack, shape [B, C_last, 8, H, W].
        Useful if you want to inspect equivariance before orientation pooling.
        """
        x = self.lift(x)                 # [B, C0, 8, H, W]
        x = self.lift_bn(x)
        x = self.lift_relu(x)
        if self.lift_pool is not None:
            x = self.lift_pool(x)
        x = self.blocks(x)               # [B, C_last, 8, H', W']
        return x

    def forward(self, x):
        x = self.forward_features(x)     # [B, C_last, 8, H, W]
        x = self.gpool(x)                # [B, C_last, H, W]  (invariant over p4m)
        x = self.head_gap(x).squeeze(-1).squeeze(-1)  # [B, C_last]
        x = self.head_fc(x)              # [B, num_classes]
        return x


# ---------------------------
# Optional: end-to-end equivariance sanity check (before orientation pooling)
# ---------------------------

@torch.no_grad()
def equivariance_check_end2end():
    torch.manual_seed(0)
    B, C, H, W = 2, 1, 32, 32
    model = P4M_EquivariantCNN(in_channels=C, channels=(8, 16), num_classes=10, pool_ks=3).eval()
    x = torch.randn(B, C, H, W)

    # features BEFORE orientation pooling (should be equivariant)
    y = model.forward_features(x)  # [B, C_last, 8, H, W]

    errs = []
    for g in range(8):
        x2 = utils.transform_image_by(g, x)
        y2 = model.forward_features(x2)

        # apply induced action on y: spatial L_g + group left action h -> g^{-1} h
        y_sp = utils.transform_image_by(g, y.view(B*y.size(1)*8, 1, H, W)).view(B, y.size(1), 8, H, W)
        inv_g = utils.p4m_inv(g)
        perm = torch.tensor([utils.p4m_compose(inv_g, h) for h in range(8)], device=y.device)
        y1g = y_sp.index_select(dim=2, index=perm)

        errs.append((y1g - y2).abs().max().item())
    print("End2End equivariance (features) max |Δ|:", max(errs))

def p4m_cnn():
    return P4M_EquivariantCNN(in_channels=1, channels=(16,32,64), num_classes=10, pool_ks=3)

if __name__ == "__main__":
    equivariance_check_end2end()
    # Example forward:
    # m = P4M_EquivariantCNN(in_channels=1, channels=(16,32,64), num_classes=10, pool_ks=3)
    # x = torch.randn(8, 1, 28, 28)
    # logits = m(x)
