import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from typing import Optional, List, Iterable, List

def get_activation(name: str):
    return {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}.get(name, nn.ReLU)

class MLPClassifier(nn.Module):
    """
    Simple MLP that outputs class logits.
    """
    def __init__(
            self,
            input_dim: int = 28*28, 
            hidden: List[int] = [256, 128], 
            activation: str = 'relu',
            dropout: float = 0.0,
            num_classes: int = 10,
            use_batchnorm: bool = False,
            final_bias: float = 0.0
    ) -> None:
        super().__init__()
        act = get_activation(activation)
        layers : List[nn.Module] = []
        prev = input_dim

        layers.append(nn.Flatten())

        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        # final linear to num_classes logits
        self.head = nn.Linear(prev, num_classes)
        nn.init.constant_(self.head.bias, final_bias)

        self.backbone = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Xavier init for linear layers; BN left as default.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        returns logits: [B, num_classes]
        """
        x = self.backbone(x)
        logits = self.head(x)
        return logits
    
    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        returns preds: [B, num_classes] (logits)
        """
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=-1)
        return preds

def mlp():
    return MLPClassifier()