import torch
from torch import nn
import torch.nn.functional as F


# SwiGLU激活：silu(gate) * up
# 输入是gate_up_proj的输出，最后一维是[gate, up]的拼接，前一半是gate，后一半是up
# silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + e^-x)
# silu类似ReLU但更平滑，x很大时silu(x)=x，x很小时，silu(x)=0
# gate经过silu后作为们，逐个元素控制up的每个分量通过多少
#
#   gate_up [N, 2*I/tp]
#       | chunk
#       |--> gate [N, I/tp] --> silu() ──┐
#       └──> up   [N, I/tp] ─────────────┴──> 逐元素相乘 --> [N, I/tp]
#
# 1) 实际用法：在Qwen3MLP 中（qwen3-0.6B，tp=1，H=1024，I=3072）
#       HF写法：    down_proj(silu(gate_proj(x)) * up_proj(x))  gate_proj、up_proj 是两次矩阵乘法
#       nano-vllm： down_proj(SiluAndMul(gate_up_proj(x)))      合并成一次矩阵乘法，再切开
#       x [N, 1024] -> gate_up_proj -> [N, 6144] -> SiluAndMul -> [N, 3072] -> down_proj -> [N, 1024]
#                                            ↑ 前3072列是gate，后3072列是up
#
# 2）怎么切：chunk(2, -1)沿最后一维从正中间一刀切开
#       gate_up 的一行：[ g0  g1  ...  g3071 │ u0  u1  ...  u3071 ]
#                        └─── x（gate） ──┘   └─── y（up） ────┘
#    
# 张量并行:
#   本模块没有参数, 不需要切分, 代码与 tp 无关.
#   切分发生在前面的 gate_up_proj (MergedColumnParallelLinear): 每张卡只加载
#   本卡神经元的 gate 行和 up 行, 拼成 [gate_local; up_local]. 因此每张卡上的
#   输入仍然是 "前一半 gate, 后一半 up, 位置一一对应", 只是变窄了:
#
#   tp=2, card 0:  [ g0  g1 | u0  u1 ]  --chunk-->  (g0,u0) (g1,u1)
#   tp=2, card 1:  [ g2  g3 | u2  u3 ]  --chunk-->  (g2,u2) (g3,u3)
#
#   silu 和乘法都是逐元素的, 神经元之间互不依赖, 两张卡各算各的, 无需通信.
#
# x: [N, 2*I/tp] -> [N, I/tp]  (I = intermediate_size)
class SiluAndMul(nn.Module):

    # torch.compile 把 chunk、silu、乘法融合成一个 kernel, 减少显存读写
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chunk 返回视图, 不拷贝: gate [N, I/tp], up [N, I/tp]
        x, y = x.chunk(2, -1)
        # 逐元素: [N, I/tp]
        return F.silu(x) * y
