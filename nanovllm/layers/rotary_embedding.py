from functools import lru_cache
import torch
from torch import nn

# RoPE全局视角：head_dim=128，base=1000000，max_pos=40960
# 类比：把q、k的128个数两两配对成64对，每一对看成时钟上的一根指针
#       token在第p个位置，就把每根指针转到“p时刻”的角度
#       这样注意力比较q、k时，就知道他们相隔多远了 
# 1) 全流程：
#   ┌─ 启动时，只算一次（RotaryEmbedding.__init__）
#   │
#   │   get_rope(128, 128, 40960, 1000000)      28层都调用，但 lru_ache 装饰器让他们共用一个对象
#   │     ├─ inv_freq [64]          64根指针各自的转速  
#   │     ├─ t [40960]              所有位置 0,1,2,3,...,40959
#   │     ├─ freqs [40960, 64]      freqs[p][i] = p * inv_freq[i] = 位置p第i根指针转到的角度
#   │     ├─ cos、sin [40960， 64]   每个角度的 cos和sin
#   │     ├─ cos_sin_cache [40960, 1, 128]  每行 = [64个cos | 64个sin]，存成buffer，之后直接查表
#   │
#   └──────────────────────────────────────────────────────────────────────────────
#
#   ┌─ 每次前向、每一层（Qwen3Attention.forward中，qk_norm之后，注意力计算之前）
#   │   
#   │   输入：positions [N] 每个token在自己序列中的位置，例如 [0 1 2 3 4 | 4 5 6]
#   │           q [N, 16, 128], k [N, 8, 128]
#   │     │
#   │     ▼  查表：cos_sin_cache[positions]
#   │   cos_sin [N, 1, 128] 切成两半 -> cos [N, 1, 64], sin [N, 1, 64] 每个token拿到所在位置的64个角度
#   │     │
#   │     ▼  apply_rotary_emb(q, cos, sin), k同样处理
#   │   q的128个数切成前64个(x1)、后64个(x2)，第i对 = (x1[i], x2[i])，转动角度 = p * inv_freq[i]
#   │   转完拼回[N, 16, 128]
#   │     │
#   │     ▼
#   │   旋转后的 k 写进 KV cache（以后再用不用重转）；旋转后的 q、k 进入注意力计算
#   │
#   └──────────────────────────────────────────────────────────────────────────────
#
# 2）用时钟理解“频率”：64对维度=64根转速不同的指针，token的位置=时间
#       第i根指针的转速 = inv_freq[i] = 1000000^(-2i/128) 位置p时的角度 = p * 转速
# 
#       指针    对应维度对  每走1个token转多少  转一圈要多少token
#       ──────────────────────────────────────────────────────────────────────────────
#       秒针    第0对       57.3°               约6个
#       ...
#       分针    第32对      0.057°              约6283个
#       时针    第63对      0.00007°            约506万
#       
#       各管一段距离：
#           秒针：相邻token差57°，一眼分清前一个还是后一个，但转的太快，位置6已经转了344°，和位置0只差16°，距离远一点就分不清前后了
#           分针：相邻token几乎不动，但几百几千token之后明细不同，负责分辨“大概隔了几百几千”
#           时针：整个上下文转不满一圈，永远不会“绕回来”混淆，负责分辨最远的距离
#       就像看表，只看秒针不知道几点，只看时针不知道几分几秒，64根针一起看，每个位置都独一无二
#       base越大，指针越慢，能区分的距离越长
#
# 3）为什么这样做能让注意力感知相对距离
#
#     注意力分数来自 q*k，对每一对维度，位置对q*k的影响只体现在q指针和k指针的夹角上
#         夹角 = (q的位置-k的位置) * 转速
#     就像判断两个时刻隔了多久，只要看指针之间差了多少角度，跟"现在是几点"无关
#     所以位置 3 看位置 1、位置 103 看位置 101，夹角一样，注意力感受到的都是"往前隔了2个"
#
# 对x应用旋转位置编码
# 把head_dim分成前后两半(x1, x2)，第i维与第i+D/2维组成一对，按角度θ旋转：
#   y1 = x1·cosθ - x2·sinθ
#   y2 = x2·cosθ + x1·sinθ
# 旋转后q*k的内积只与两者相对位置差有关，从而实现相对位置编码
#
# 二维直观理解：把一对数(x1, x2)看做平面内的一个点，乘以旋转矩阵就是把它绕原点旋转θ度
#   [y1]   [cosθ  -sinθ] [x1]
#   [y2] = [sinθ   cosθ] [x2]
#   位置为m的token转 m·θ 角，位置为n的转 n·θ 角；两个向量的内积只取决于夹角 (m-n)·θ
#   所以注意力分数只与相对位置有关
#
# 例子：x=[a, b, c, d] -> x1 = [a, b], x2 = [c, d]
# 分组为(a, c)和(b, d)，分别按角度θ0、θ1 旋转
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 沿最后一维平均切成2块
    # [N, heads, head_dim] -> [N, heads, head_dim/2]
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 沿最后一维拼接
    # [N, heads, head_dim/2] -> [N, heads, head_dim]
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


# 预先算好所有位置的cos/sin，前向时直接按位置查表
# cos_sin_cache的结构和查表过程（D=128，每个头64个维度对，f_i = inv_freq[i]
#   cos_sin_cache [max_pos, 1, 128]
#              ├────── 前64个：cos ──────┤  ├────── 后64个：sin ──────┤
#       位置0    cos(0·f0) ... cos(0·f63)   sin(0·f0)  ...  sin(0·f63)
#       位置1    cos(1·f0) ... cos(1·f63)   sin(1·f0)  ...  sin(1·f63)
#       ...
#       位置m    cos(m·f0) ... cos(m·f63)   sin(m·f0)  ...  sin(m·f63)
#   forward：
#       positions [N] --查表--> cos_sin [N, 1, 128] --chunk--> cos [N, 1, 64], sin [N, 1, 64]
#       q [N, Hq, 128] --chunk--> x1 = 前64维，x2=后64维 [N, Hq, 64]
#                        --旋转--> y1 = x1*cos - x2*sin, y2=x2*cos + x1*sin (cos/sin在Hq个头上广播)
#                        --cat--> [N, Hq, 128]
#       k同理（Hkv个头）；v不做旋转
#
# 为什么旋转后注意力只和相对距离有关（二维、每个位置转1弧度的小例子）：
#   q = [1, 0] 在位置3 -> 转3弧度 -> [cos3, sin3]
#   k = [1, 0] 在位置1 -> 转1弧度 -> [cos1, sin1]
#   q·k = cos3·cos1 + sin3·sin1 = cos(3 - 1) = cos2
#   把两者都往后挪2个位置（5和3），q·k = cos(5 - 3) = cos2，结果不变：只取决于距离 2
class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        # 每个头的维度
        self.head_size = head_size
        # 只支持对整个head_dim做旋转
        assert rotary_dim == head_size
        # 每一对维度的旋转频率：inv_freq_i = base^(-2i/D), i=0,1,2,3,...,D/2-1
        # inv_freq: [D/2]
        # torch.arrange(0, D, 2): [0, 2, 4, ..., D-2]，共D/2个数
        # 例如 D=4，base=10000：
        #   arange = [0, 2], /D = [0, 0.5], base^ = [1, 100]，倒数=[1, 0.01]
        #   第0对每个位置旋转1弧度，第1对每个位置旋转0.01弧度
        # 低维旋转快，捕捉近距离关系；高维度旋转慢，捕捉远距离关系
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 所有位置 [max_pos]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：freqs[pos, i] = pos * inv_freq_i，即位置pos在第i对维度上的旋转角度
        # [max_pos] ⊗ [D/2] -> [max_pos, D/2]
        # torch.einsum：用字符串描述张量运算，等价于t.unsqueeze(1) * inv_freq.unsqueeze(0)，即[max_pos, 1] * [1, D/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # [max_pos, D/2] -> [max_pos, D] -> [max_pos, 1, D]
        # 中间的1用于在q/k [N, heads, D]上广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # register_buffer：注册一个张量，跟随模块一起移动，但不是可训练参数
        # persistent=False：不保存到state_dict，加载权重时也不需要它
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # positions [N]；query：[N, Hq, D]; key: [N, Hkv, D]
    # 返回旋转后的query [N, Hq, D], key [N, Hkv, D]
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按每个token在序列中的位置取出对应的cos/sin：[max_pos, 1, D] -> [N, 1, D]
        # 用张量当下标，一次取出多行：positions=[0, 1, 2, 0, 1] -> 取第0、1、2、0、1行
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# lru_cache：参数相同时，返回同一个实例，所有层共享一份cos/sin缓存，节省显存
# lru_cache(n)会记住最近n组参数对应的返回值
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
