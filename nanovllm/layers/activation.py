import torch
from torch import nn
import torch.nn.functional as F


# SwiGLU激活：silu(gate) * up
# 输入是gate_up_proj的输出，最后一维是[gate, up]拼接，前一半是gate，后一半是up
# silu(x) = x * sigmoid(x)
class SiluAndMul(nn.Module):

    # torch.compile把chunk、silu、乘法融合成一个kernel，减少显存读写
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
