import torch
from torch import nn


# RMSNorm：y = x / sqrt(mean(x^2) + eps) * weight
# 相比LayerNorm，不减均值、没有bias，计算更省，效果相当
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # 防止除零
        self.eps = eps
        # 可学习的逐通道缩放系数
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 单纯的RMSNorm
    # torch.compile把多个逐元素操作融合成一个kernel
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        # 转成float32计算，bf16/fp16下平方求和容易损失精度
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt = 1 / sqrt，原地操作减少显存分配
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    # 残差相加 + RMSNorm融合：
    #   residual = x + residual（作为下一个子层的残差）
    #   x = RMSNorm(residual)（作为下一个子层的输入）
    # 融合后少一次读写hidden_states，降低访存开销
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    # 传入residual时走融合版本，返回(归一化结果, 新残差)
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
