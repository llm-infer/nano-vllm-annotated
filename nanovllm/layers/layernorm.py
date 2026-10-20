import torch
from torch import nn


# RMSNorm：y = x / sqrt(mean(x^2) + eps) * weight
# 相比LayerNorm，不减均值、没有bias，计算更省，效果相当
# 在最后一维上归一化，前面的维度任意：
#   decoder层中 x: [N, H]，weight: [H]
#   QK-Norm中   x: [N, heads, D]，weight: [D]（每个头独立归一化）
#
# 数值例子（H=4，eps忽略，weight全为1）：
#   x            = [1, 2, 3, 4]
#   x^2          = [1, 4, 9, 16]
#   mean(x^2)    = 30 / 4 = 7.5
#   sqrt(7.5)    ≈ 2.739
#   y = x / 2.739 ≈ [0.365, 0.730, 1.095, 1.461]
#   归一化后向量的"均方根"为1，不管输入数值多大，输出的尺度都稳定
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # 加在均方值上，防止除零
        self.eps = eps
        # 可学习的逐通道缩放系数，weight: [hidden_size]，初始化为1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 单纯的RMSNorm，x: [..., hidden_size] -> [..., hidden_size]
    #   形状中的 ... 表示"前面任意多个维度"
    # torch.compile把多个逐元素操作融合成一个kernel
    #   不融合时：pow、mean、rsqrt、mul……每一步都要从显存读一遍、写一遍
    #   融合后：一次读入，在GPU寄存器里算完所有步骤，一次写出，访存次数大幅减少
    #   第一次调用时会编译（耗时几秒~几十秒），之后复用编译结果
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        # 转成float32计算，bf16/fp16下平方求和容易损失精度
        x = x.float()
        # 均方值：[..., hidden_size] -> [..., 1]（keepdim保留维度，便于广播）
        #   x.pow(2)：逐元素平方
        #   mean(dim=-1)：对最后一维求平均；keepdim=True 时该维度保留为1，否则直接去掉
        #     [N, H].mean(-1)               -> [N]
        #     [N, H].mean(-1, keepdim=True) -> [N, 1]（可以直接和[N, H]广播相乘）
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt = 1 / sqrt，[..., 1]广播到[..., hidden_size]；原地操作减少显存分配
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原dtype，乘以weight（[hidden_size]广播到最后一维）
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    # 残差连接：子层输出 + 子层输入，让信息和梯度可以"跳过"子层直接传递，是深层网络能训练的关键
    # 残差相加 + RMSNorm融合：
    #   residual = x + residual（作为下一个子层的残差）
    #   x = RMSNorm(residual)（作为下一个子层的输入）
    # x、residual: [N, H] -> 返回 (x [N, H], residual [N, H])
    # 融合后少一次读写hidden_states，降低访存开销
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 在float32下相加
        x = x.float().add_(residual.float())
        # 新的残差保存为原dtype
        residual = x.to(orig_dtype)
        # [N, H] -> [N, 1]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    # 传入residual时走融合版本，返回(归一化结果, 新残差)；否则只返回归一化结果
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
