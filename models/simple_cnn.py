import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from typing import Optional, List, Iterable, List

def get_activation(name: str):
    return {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}.get(name, nn.ReLU)

class SimpleCNN(nn.Module):
    """
    Minimal CNN classifier with global average pooling head.
    Works for grayscale (C=1) or RGB (C=3) by setting in_channels.
    """
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [32, 64, 128],
        num_classes: int = 10,
        activation: str = "relu",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        Act = get_activation(activation)
        feats = []
        c_prev = in_channels
        for c in channels:
            feats += [nn.Conv2d(c_prev, c, kernel_size=3, padding=1, bias=not use_batchnorm)]
            if use_batchnorm:
                feats += [nn.BatchNorm2d(c)]
            feats += [Act(inplace=True) if "inplace" in Act.__init__.__code__.co_varnames else Act()]
            # Downsample
            feats += [nn.MaxPool2d(kernel_size=2)]  # H,W 반으로
            if dropout > 0:
                feats += [nn.Dropout2d(p=dropout)]
            c_prev = c
        self.backbone = nn.Sequential(*feats)

        # Global Average Pooling and final linear layer
        # GAP: [B, C, H, W] -> [B, C, 1, 1]
        # Head: [B, C, 1, 1] -> [B, num_classes]
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.head = nn.Linear(channels[-1], num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        # He initialize Conv2d and Linear layers
        # Biases are initialized to zero.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns logits: [B, num_classes]
        """
        x = self.backbone(x)
        x = self.gap(x).flatten(1)  # [B, C]
        logits = self.head(x)
        return logits

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=-1)

def simpleCNN():
    return SimpleCNN()