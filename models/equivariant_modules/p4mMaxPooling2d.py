import torch
import torch.nn as nn
import torch.nn.functional as F
import models.equivariant_modules.utils as utils

from models.equivariant_modules.p4mConv2d import P4M_LiftConv2d

class P4M_MaxPool2d(nn.Module):
    """
    Group-equivariant spatial max pooling for p4m features.

    Input : [B, C, 8, H, W]  or  [B, C, H, W]
    Output: [B, C, 8, H2, W2] or  [B, C, H2, W2]

    - stride는 항상 1 (subsampling 없음)
    - 각 그룹 성분에 동일한 2D max-pool 적용 → 등변성 유지
    """
    def __init__(self, kernel_size=2, padding=0, ceil_mode=False):
        super().__init__()
        if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
        else: self.kernel_size = kernel_size
        if isinstance(padding, int): self.padding = (padding, padding)
        else: self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # 일반 텐서 [B,C,H,W] -> 표준 non-strided MaxPool2d
            return F.max_pool2d(
                x, kernel_size=self.kernel_size, stride=1,
                padding=self.padding, ceil_mode=self.ceil_mode
            )

        assert x.dim() == 5 and x.size(2) == 8, "Expected [B, C, 8, H, W]"
        B, C, G, H, W = x.shape

        # (중요) h-주도로 flatten → [B, 8*C, H, W]
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B, G * C, H, W)
        y_flat = F.max_pool2d(
            x_flat, kernel_size=self.kernel_size, stride=1,
            padding=self.padding, ceil_mode=self.ceil_mode
        )
        _, _, H2, W2 = y_flat.shape
        y = y_flat.view(B, G, C, H2, W2).permute(0, 2, 0+1, 3, 4).contiguous()  # back to [B,C,8,H2,W2]
        return y


class P4M_AvgPool2d(nn.Module):
    """평균 풀링 버전(등변성 동일)."""
    def __init__(self, kernel_size=2, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__()
        if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
        else: self.kernel_size = kernel_size
        if isinstance(padding, int): self.padding = (padding, padding)
        else: self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return F.avg_pool2d(
                x, kernel_size=self.kernel_size, stride=1,
                padding=self.padding, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad
            )
        assert x.dim() == 5 and x.size(2) == 8, "Expected [B, C, 8, H, W]"
        B, C, G, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B, G * C, H, W)
        y_flat = F.avg_pool2d(
            x_flat, kernel_size=self.kernel_size, stride=1,
            padding=self.padding, ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad
        )
        _, _, H2, W2 = y_flat.shape
        return y_flat.view(B, G, C, H2, W2).permute(0, 2, 0+1, 3, 4).contiguous()


class P4M_OrientationPool(nn.Module):
    """
    Group pooling over the group axis -> invariant.
    mode='max' | 'avg' | 'l2'
    """
    def __init__(self, mode='avg'):
        super().__init__()
        assert mode in ('max', 'avg', 'l2')
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5 and x.size(2) == 8, "Expected [B, C, 8, H, W]"
        if self.mode == 'max':
            return x.max(dim=2).values
        elif self.mode == 'avg':
            return x.mean(dim=2)
        else:  # l2
            return torch.linalg.vector_norm(x, dim=2)


# ------------------ Quick equivariance check ------------------
@torch.no_grad()
def quick_check_pool():
    torch.manual_seed(0)
    B, Cin, H, W = 2, 3, 32, 32  # 정사각형 권장
    x = torch.randn(B, Cin, H, W)

    # 네트워크 표현과 동일한 방식으로: lift → pool
    lift = P4M_LiftConv2d(Cin, 6, kernel_size=3, padding=1)
    pool = P4M_MaxPool2d(kernel_size=3, padding=1)  # stride=1 고정

    f  = lift(x)         # [B, 6, 8, H, W]
    y  = pool(f)         # [B, 6, 8, H, W]

    errs = []
    for g in range(8):
        x2 = utils.transform_image_by(g, x)
        f2 = lift(x2)
        y2 = pool(f2)

        # L_g ∘ y  (공간 변환) + 그룹축은 h ↦ g^{-1} h (좌작용)
        y_sp = utils.transform_image_by(g, y.view(B*6*8, 1, H, W)).view(B, 6, 8, H, W)
        inv_g = utils.p4m_inv(g)
        perm = torch.tensor([utils.p4m_compose(inv_g, h) for h in range(8)], device=y.device)
        y1g = y_sp.index_select(dim=2, index=perm)

        errs.append((y1g - y2).abs().max().item())
    print("P4M_MaxPool2d equivariance max |Δ|:", max(errs))

if __name__ == "__main__":
    quick_check_pool()