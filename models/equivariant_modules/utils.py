import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
---------------------------
p4m (D4) group utilities
---------------------------
We encode group elements as indices 0..7:
0..3: rotations r^k (k=0,1,2,3)
4..7: reflections m r^k (k=0,1,2,3), where m is a mirror across the vertical axis (left-right flip)

Convention for transform_by(g): first rotate by k*90° (CCW), then (if reflection) apply left-right flip.
With this convention, composition matches our compose() below.
'''

def _to_pair(g: int):
    if g < 4: return False, g
    return True, g - 4

def _from_pair(is_ref: bool, k: int):
    k %= 4
    return (4 + k) if is_ref else k

def p4m_compose(g1: int, g2: int) -> int:
    ref1, k1 = _to_pair(g1); ref2, k2 = _to_pair(g2)
    if not ref1 and not ref2:   # r^k1 ∘ r^k2
        return _from_pair(False, k1 + k2)
    if ref1 and not ref2:       # (m r^k1) ∘ r^k2
        return _from_pair(True, k1 + k2)
    if not ref1 and ref2:       # r^k1 ∘ (m r^k2)
        return _from_pair(True, k2 - k1)
    return _from_pair(False, k2 - k1)  # (m r^k1) ∘ (m r^k2)

def p4m_inv(g: int) -> int:
    ref, k = _to_pair(g)
    return _from_pair(False, -k) if not ref else _from_pair(True, k)

def transform_image_by(g: int, x: torch.Tensor) -> torch.Tensor:
    ref, k = _to_pair(g)
    y = torch.rot90(x, k=k, dims=(-2, -1))
    if ref: y = torch.flip(y, dims=(-1,))
    return y

def transform_kernel_by(g: int, w: torch.Tensor) -> torch.Tensor:
    # same rule as image: L_g w
    ref, k = _to_pair(g)
    y = torch.rot90(w, k=k, dims=(-2, -1))
    if ref: y = torch.flip(y, dims=(-1,))
    return y
