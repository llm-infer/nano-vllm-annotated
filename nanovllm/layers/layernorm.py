import torch
from torch import nn


# RMSNorm：y = x / sqrt(mean(x^2) + eps) * weight
# 把一组数据伸缩到一个特定范围里，让计算更加稳定；归一化后这组数据的均方根为1，不管输入数值多大，输出都很稳定
# 相比layernorm，不减均值，没有bias，计算更快
# 在最后一维做归一化，前面的维度任意
# decoder层：x [N, H], weight [H]
# QK-Norm：x [N, heads, head_dim], weight [head_dim]（每个头独立归一化）
#
# 数值例子：H=4，eps忽略，weight全为1
#   x           = [1, 2, 3, 4]
#   x^2         = [1, 4, 9, 16]
#   mean(x^2)   = 7.5
#   sqrt(7.5)   = 2.739
#   y           = x / 2.739 = [0.365, 0.73, 1.095, 1.461]
#
# 实际用法：qwen3每层4个RMSNorm，所有layers之后1个RMSNOrm
#     模型中的位置                 输入x          weight      是否带residual
#     ──────────────────────────────────────────────────────────────────────────────────
#     input_layernorm            [N, 1024]      [1024]     除了第1层，其他都带
#     post_attentioin_layernorm  [N, 1024]      [1024]          是
#     self_attn.q_norm           [N, 16, 128]   [128]           否
#     self_attn.k_norm           [N, 8, 128]    [128]           否
#     model.norm                 [N, 1024]      [1024]          是
# q_norm/k_norm的输入是3维，但是只在最后一维做归一化，相当于每个token的注意力都各自做归一化
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # 加载均方值上，防止除零
        self.eps = eps
        # 可学习的缩放系数，初始化为1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # torch.compile把多个element-wise操作合并成一个kernel（优化访存次数）
    #   不融合时：pow、mean、rsqrt、mul每一步都要从显存中读出来，计算，再写回显存
    #   融合后：一次读入，在GPU寄存器中完成所有操作，最后写回显存，访存次数大幅减少
    # 第一次调用会编译，之后复用编译结果
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        # 转为float32，bf16/fp16平方求和很容易损失精度、溢出
        # 浮点数指数决定范围，尾数决定精度
        # bf16和fp32指数位都是8，范围一样大，但是精度太粗，容易损失精度
        # fp16指数位是5，数值范围6.1*10^5 ~ 65504，范围太小，容易溢出
        # bf16和fp32分别适用什么场景
        #   矩阵乘：乘一次，误差只是一点点比例。可用bf16
        #       a' × b' = a(1 + ε₁) × b(1 + ε₂)
        #               = a × b × (1 + ε₁)(1 + ε₂)
        #               = a × b × (1 + ε₁ + ε₂ + ε₁ε₂)
        #       相对误差 = ε₁ + ε₂ + ε₁ε₂，与a、b无关，误差可控
        #   加法：误差累积，随着累加次数成比例放大。对精度敏感，需要用fp32
        #       a' + b' = a + aε₁ + b + bε₂
        #               = (a + b) + (aε₁ + bε₂)
        #       相对误差 = (aε₁ + bε₂) / (a + b)
        #       a、b 没有被约掉。误差的大小取决于 a、b 本身
        x = x.float()
        # 在最后一维求均方值：[..., hidden_size] -> [..., 1], keepdim保留维度，便于广播
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt = 1/sqrt
        # [..., 1] 广播到 [..., hidden_size]
        # mul_原地操作，减少显存分配
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原type，乘以weight
        # [..., hidden_size] * [hidden_size]：广播到最后一维，最后一维元素逐个相乘
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    # 残差连接：子层输出 + 子层输入，让信息可以跳过子层直接传递，是深层网络能够训练的关键
    # 残差连接 + RMSNorm融合：
    #   residual = x + redidual（作为下一个子层的残差）
    #   x = RMSNorm(residual)（作为下一个子层的输入）
    #
    #   x（上一个子层的输出）────┐
    #                        (+) ──┬──>新的residual───────> 返回值2：留给下一个子层做残差
    #   residual ─────────────┘    │
    #                              └──> RMSNorm ──> × weight ──> 返回值1：作为下一个子层的输入
    #
    # 融合后少一次hidden_states读写，降低访存开销
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 使用float32相加
        x = x.float().add_(residual.float())
        # 新的残差保存为原type
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    # 什么时候residual为None
    #   第0层的input_layernorm：前面没有残差，embedding输出直接作为残差
    #   QK-Norm：只对q、k做归一化，和残差
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
