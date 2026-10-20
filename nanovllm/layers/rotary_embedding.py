from functools import lru_cache
import torch
from torch import nn


# 对x应用旋转位置编码（rotate-half / GPT-NeoX风格）
# 把head_dim分成前后两半(x1, x2)，第i维与第i+D/2维组成一对，按角度θ旋转：
#   y1 = x1·cosθ - x2·sinθ
#   y2 = x2·cosθ + x1·sinθ
# 旋转后q·k的内积只与两者的位置差有关，从而实现相对位置编码
# x: [N, heads, D]；cos、sin: [N, 1, D/2]（在heads维上广播）
# 返回: [N, heads, D]
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 在float32下计算，保证三角函数乘法的精度
    # [N, heads, D] -> x1, x2 各 [N, heads, D/2]
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # [N, heads, D/2] * [N, 1, D/2] -> [N, heads, D/2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼回 [N, heads, D]，转回原dtype
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


# 旋转位置编码（RoPE），预先计算好所有位置的cos/sin，前向时按位置查表
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
        # 低维旋转快（捕捉近距离关系），高维旋转慢（捕捉远距离关系）
        # base越大，最低频率越低，能区分的上下文越长（Qwen3的base=1000000）
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 所有位置：t: [max_pos]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：freqs[pos, i] = pos * inv_freq_i，即位置pos在第i对维度上的旋转角度
        # [max_pos] ⊗ [D/2] -> [max_pos, D/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()  # [max_pos, D/2]
        sin = freqs.sin()  # [max_pos, D/2]
        # 拼接：[max_pos, D]（前一半cos，后一半sin） -> unsqueeze(1) -> [max_pos, 1, D]
        # 中间的1用于在头维度上广播（q/k形状为[N, heads, D]）
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
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
        cos_sin = self.cos_sin_cache[positions]
        # [N, 1, D] -> cos、sin 各 [N, 1, D/2]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# lru_cache：参数相同时返回同一个实例，所有层共享一份cos/sin缓存，节省显存
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
