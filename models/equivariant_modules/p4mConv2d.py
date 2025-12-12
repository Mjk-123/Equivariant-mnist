import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.equivariant_modules.utils as utils


# ---------------------------
# Z^2 -> p4m lifting conv
# ---------------------------

class P4M_LiftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int): kH = kW = kernel_size
        else: kH, kW = kernel_size
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kH, kW) *
                                   (1.0 / math.sqrt(in_channels * kH * kW)))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        outs = []
        for g in range(8):
            w_g = utils.transform_kernel_by(g, self.weight)  # L_g ψ
            y_g = F.conv2d(x, w_g, bias=self.bias, stride=self.stride,
                           padding=self.padding, dilation=self.dilation)
            outs.append(y_g.unsqueeze(2))
        return torch.cat(outs, dim=2)  # [B, C_out, 8, H', W']

# ---------------------------
# p4m -> p4m group conv (reference implementation)
# ---------------------------
    
class P4M_GroupConv2d_Ref(nn.Module):
    """
    정의 그대로의 G-correlation 구현:
      y(g) = sum_{h in p4m}  conv2d( x(h),  L_g ψ( g^{-1} h ) )

    - ψ(·)는 '상대 원소 u = g^{-1}h' 만 의존하도록 파라미터화: weight_rel[:, :, u]
    - L_g는 필터에 적용 (논문 7.1 'filter transformation').
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int): kH = kW = kernel_size
        else: kH, kW = kernel_size
        self.C_in, self.C_out = in_channels, out_channels
        self.kH, self.kW = kH, kW
        self.stride, self.padding, self.dilation = stride, padding, dilation
        # ψ(u): [C_out, C_in, 8, kH, kW]
        self.weight_rel = nn.Parameter(
            torch.randn(out_channels, in_channels, 8, kH, kW) * (1.0 / math.sqrt(in_channels * kH * kW))
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        # x: [B, C_in, 8, H, W]
        B, Cin, N, H, W = x.shape
        assert Cin == self.C_in and N == 8

        outs = []
        for g in range(8):
            inv_g = utils.p4m_inv(g)
            acc = None
            for h in range(8):
                u = utils.p4m_compose(inv_g, h)                      # u = g^{-1} h
                W_u = self.weight_rel[:, :, u, :, :]           # ψ(u)
                W_eff = utils.transform_kernel_by(g, W_u)            # L_g ψ(u)

                y = F.conv2d(x[:, :, h, :, :], W_eff, bias=None,
                             stride=self.stride, padding=self.padding, dilation=self.dilation)  # [B, C_out, H', W']
                acc = y if acc is None else (acc + y)
            if self.bias is not None:
                acc = acc + self.bias.view(1, -1, 1, 1)
            outs.append(acc.unsqueeze(2))  # [B, C_out, 1, H', W']
        return torch.cat(outs, dim=2)      # [B, C_out, 8, H', W']
    
# ---------------------------
# p4m -> p4m group conv (fast block-assembled version)
# ---------------------------

class P4M_GroupConv2d_Fast(nn.Module):
    """
    Fast block-assembled version matching the 'Ref' definition exactly:
      y(g) = sum_h conv2d( x(h),  L_g ψ( g^{-1} h ) )

    Input : [B, C_in, 8, H, W]
    Output: [B, C_out, 8, H_out, W_out]
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size

        self.C_in  = in_channels
        self.C_out = out_channels
        self.kH, self.kW = kH, kW
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # ψ(u): [C_out, C_in, 8, kH, kW]
        self.weight_rel = nn.Parameter(
            torch.randn(out_channels, in_channels, 8, kH, kW) / math.sqrt(in_channels * kH * kW)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        # x: [B, C_in, 8, H, W]
        B, Cin, N, H, W = x.shape
        assert Cin == self.C_in and N == 8, "Expected [B, C_in, 8, H, W]"

        # Flatten group into channels once: [B, C_in*8, H, W]
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B, 8 * Cin, H, W)

        outs = []
        for g in range(8):
            inv_g = utils.p4m_inv(g)

            # Assemble big kernel for this g: [C_out, C_in*8, kH, kW]
            K_concat = x_flat.new_zeros((self.C_out, Cin * 8, self.kH, self.kW))

            for h in range(8):
                u = utils.p4m_compose(inv_g, h)                     # u = g^{-1} h
                W_u = self.weight_rel[:, :, u, :, :]          # ψ(u): [C_out, C_in, kH, kW]
                W_eff = utils.transform_kernel_by(g, W_u)           # L_g ψ(u)   <-- 중요!

                # place into block for input-group channel h
                K_concat[:, h*Cin:(h+1)*Cin, :, :] = W_eff

            y_g = F.conv2d(x_flat, K_concat, bias=self.bias,
                           stride=self.stride, padding=self.padding, dilation=self.dilation)
            outs.append(y_g.unsqueeze(2))  # [B, C_out, 1, H', W']

        return torch.cat(outs, dim=2)      # [B, C_out, 8, H', W']
    
class P4M_BatchNorm(nn.Module):
    """
    Shared-parameter BatchNorm over group components.
    We reshape [B, C, 8, H, W] -> [B*8, C, H, W], apply BN2d(C), then reshape back.
    This applies the same affine params to every group slice and mixes stats over (B*8).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                                 affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if x.dim() == 4:
            return self.bn(x)  # fallback
        B, C, G, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).contiguous().view(B*G, C, H, W)
        y = self.bn(y)
        return y.view(B, G, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

class P4M_ReLU(nn.Module):
    """Pointwise nonlinearity acts identically on each group slice → equivariance preserved."""
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return self.relu(x)

# ---------- 등변성 체크 (좌작용: 공간 변환 + 그룹축 h -> g h) ----------
@torch.no_grad()
def _permute_group_and_spatial_LEFT(y: torch.Tensor, g: int) -> torch.Tensor:
    B, C, N, H, W = y.shape
    # 공간 변환: L_g
    y_sp = utils.transform_image_by(g, y.view(B*C*N, 1, H, W)).view(B, C, N, H, W)
    # 그룹축: h -> g h (좌작용)  [Lu f](h) = f(g^{-1} h) 이므로 index는 g h 가 맞습니다.
    inv_g = utils.p4m_inv(g)
    idx = torch.tensor([utils.p4m_compose(inv_g, h) for h in range(8)], device=y.device)
    return y_sp.index_select(dim=2, index=idx)

@torch.no_grad()
def quick_check(device="cpu"):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, device=device)

    lift = P4M_LiftConv2d(3, 4, kernel_size=3, padding=1).to(device)
    gconv = P4M_GroupConv2d_Ref(4, 5, kernel_size=3, padding=1).to(device)
    relu = nn.ReLU()

    y1 = gconv(relu(lift(x)))  # [B, 5, 8, H, W]

    for g in range(8):
        x2 = utils.transform_image_by(g, x)          # 입력에 L_g
        y2 = gconv(relu(lift(x2)))

        y1g = _permute_group_and_spatial_LEFT(y1, g)
        err = (y1g - y2).abs().max().item()
        print(f"g={g} | max |Δ| = {err:.3e}")

@torch.no_grad()
def compare_ref_and_fast():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 17, 19)

    lift = P4M_LiftConv2d(4, 6, kernel_size=3, padding=1)
    gref = P4M_GroupConv2d_Ref(6, 5, kernel_size=3, padding=1)
    gfast = P4M_GroupConv2d_Fast(6, 5, kernel_size=3, padding=1)

    # copy params so both use identical weights
    gfast.weight_rel.data.copy_(gref.weight_rel.data)

    z = lift(x)            # [B,6,8,H,W]
    y_ref  = gref(z)
    y_fast = gfast(z)
    print("max |ref-fast| =", (y_ref - y_fast).abs().max().item())

if __name__ == "__main__":
    compare_ref_and_fast()