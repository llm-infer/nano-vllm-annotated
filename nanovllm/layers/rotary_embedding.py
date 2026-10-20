from functools import lru_cache
import torch
from torch import nn


# 对x应用旋转位置编码（rotate-half / GPT-NeoX风格）
# 把head_dim分成前后两半(x1, x2)，第i维与第i+d/2维组成一对，按角度θ旋转：
#   y1 = x1·cosθ - x2·sinθ
#   y2 = x2·cosθ + x1·sinθ
# 旋转后q·k的内积只与两者的位置差有关，从而实现相对位置编码
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 在float32下计算，保证三角函数乘法的精度
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
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
        self.head_size = head_size
        # 只支持对整个head_dim做旋转
        assert rotary_dim == head_size
        # 每一对维度的旋转频率：inv_freq_i = base^(-2i/d)，i = 0, 1, ..., d/2-1
        # 低维旋转快（捕捉近距离关系），高维旋转慢（捕捉远距离关系）
        # base越大，最低频率越低，能区分的上下文越长（Qwen3的base=1000000）
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：freqs[pos, i] = pos * inv_freq_i，即位置pos在第i对维度上的旋转角度
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 形状：[max_pos, 1, rotary_dim]，前一半是cos，后一半是sin
        # 中间的1用于在头维度上广播（q/k形状为[N, num_heads, head_dim]）
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # persistent=False：不保存到state_dict，加载权重时也不需要它
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按每个token的位置取出对应的cos/sin：[N, 1, rotary_dim]
        cos_sin = self.cos_sin_cache[positions]
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
