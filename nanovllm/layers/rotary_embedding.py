from functools import lru_cache
import torch
from torch import nn


# 对x应用旋转位置编码（rotate-half / GPT-NeoX风格）
# 把head_dim分成前后两半(x1, x2)，第i维与第i+D/2维组成一对，按角度θ旋转：
#   y1 = x1·cosθ - x2·sinθ
#   y2 = x2·cosθ + x1·sinθ
# 旋转后q·k的内积只与两者的位置差有关，从而实现相对位置编码
#
# 二维直观理解：把一对数 (x1, x2) 看作平面上的一个点（向量），乘以旋转矩阵就是把它绕原点转θ角
#   [y1]   [cosθ  -sinθ] [x1]
#   [y2] = [sinθ   cosθ] [x2]
#   位置为m的token转 m·θ 角，位置为n的转 n·θ 角；两个向量的内积只取决于夹角 (m-n)·θ
#   所以注意力分数天然只与"相对距离"有关
#
# 为什么是前后两半配对，而不是相邻两维配对？两种写法数学上等价，只是维度排列不同，
# HF的Qwen/LLaMA权重是按"前后两半"训练的（rotate_half），所以这里必须一致
#
# 例子（D=4）：x = [a, b, c, d] -> x1 = [a, b]，x2 = [c, d]
#   配对为 (a, c) 和 (b, d)，分别按角度 θ0、θ1 旋转
# x: [N, heads, D]；cos、sin: [N, 1, D/2]（在heads维上广播）
# 返回: [N, heads, D]
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 在float32下计算，保证三角函数乘法的精度
    # [N, heads, D] -> x1, x2 各 [N, heads, D/2]
    # torch.chunk(x, 2, dim=-1)：沿最后一维平均切成2块
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # [N, heads, D/2] * [N, 1, D/2] -> [N, heads, D/2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼回 [N, heads, D]，转回原dtype
    # torch.cat((y1, y2), dim=-1)：沿最后一维拼接，[.., D/2] + [.., D/2] -> [.., D]
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


# 旋转位置编码（RoPE），预先计算好所有位置的cos/sin，前向时按位置查表
#
# cos_sin_cache 的结构与查表过程（D=128，每个头64对维度，f_i = inv_freq[i]）：
#
#   cos_sin_cache [max_pos, 1, 128]（中间的1省略不画）
#               ├────── 前64个：cos ──────┤         ├────── 后64个：sin ──────┤
#     位置0     cos(0·f0)  ...  cos(0·f63)          sin(0·f0)  ...  sin(0·f63)
#     位置1     cos(1·f0)  ...  cos(1·f63)          sin(1·f0)  ...  sin(1·f63)
#     ...
#     位置m     cos(m·f0)  ...  cos(m·f63)          sin(m·f0)  ...  sin(m·f63)
#
#   forward：
#     positions [N] ──查表──> cos_sin [N, 1, 128] ──chunk──> cos [N, 1, 64]、sin [N, 1, 64]
#     q [N, Hq, 128] ──chunk──> x1 = 前64维、x2 = 后64维
#                   ──旋转──> y1 = x1·cos - x2·sin，y2 = x2·cos + x1·sin   （cos/sin在Hq个头上广播）
#                   ──cat───> [N, Hq, 128]
#     k 同理（Hkv个头）；v 不做旋转
class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        # 每个头的维度D
        self.head_size = head_size
        # 只支持对整个head_dim做旋转
        assert rotary_dim == head_size
        # 每一对维度的旋转频率：inv_freq_i = base^(-2i/D)，i = 0, 1, ..., D/2-1
        # inv_freq: [D/2]
        # torch.arange(0, D, 2)：[0, 2, 4, ..., D-2]，共D/2个数
        # 例如 D=4，base=10000：
        #   arange = [0, 2]，/D = [0, 0.5]，base^ = [1, 100]，倒数 = [1, 0.01]
        #   第0对每个位置转1弧度（快），第1对每个位置转0.01弧度（慢）
        # 低维旋转快（捕捉近距离关系），高维旋转慢（捕捉远距离关系）
        # base越大，最低频率越低，能区分的上下文越长（Qwen3的base=1000000）
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 所有位置：t: [max_pos]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：freqs[pos, i] = pos * inv_freq_i，即位置pos在第i对维度上的旋转角度
        # [max_pos] ⊗ [D/2] -> [max_pos, D/2]
        # torch.einsum：用字符串描述张量运算，"i,j -> ij" 表示 out[i][j] = t[i] * inv_freq[j]（外积）
        #   等价写法：t.unsqueeze(1) * inv_freq.unsqueeze(0)，即 [max_pos, 1] * [1, D/2] 广播
        # 例如 t=[0,1,2]，inv_freq=[1, 0.01]：
        #   freqs = [[0, 0   ],
        #            [1, 0.01],
        #            [2, 0.02]]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()  # [max_pos, D/2]
        sin = freqs.sin()  # [max_pos, D/2]
        # 拼接：[max_pos, D]（前一半cos，后一半sin） -> unsqueeze(1) -> [max_pos, 1, D]
        # 中间的1用于在头维度上广播（q/k形状为[N, heads, D]）
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # register_buffer：登记一个"非参数"张量，会跟着模块一起移动设备（.cuda()等），但不是可训练参数
        # persistent=False：不保存到state_dict，加载权重时也不需要它
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # positions: [N]；query: [N, Hq, D]；key: [N, Hkv, D]
    # 返回旋转后的 query [N, Hq, D]、key [N, Hkv, D]
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按每个token的位置取出对应的cos/sin：[max_pos, 1, D] -> [N, 1, D]
        # 用整数张量做下标，一次取出多行：positions=[0, 1, 2, 0, 1] -> 取第0、1、2、0、1行
        cos_sin = self.cos_sin_cache[positions]
        # [N, 1, D] -> cos、sin 各 [N, 1, D/2]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# lru_cache：参数相同时返回同一个实例，所有层共享一份cos/sin缓存，节省显存
#   @lru_cache(n) 会记住最近n组参数对应的返回值，再次用相同参数调用时直接返回记住的结果，不再执行函数体
#   28层都调用 get_rope(128, 128, 40960, 1000000)，只有第一次真正创建对象
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
