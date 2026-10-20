import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# shape记号：
#   N  ：本次前向的token总数（batch内所有序列拼接成一维，没有batch维）
#   B  ：序列数
#   H  ：hidden_size
#   Hq ：当前rank的q头数 = num_attention_heads / tp
#   Hkv：当前rank的kv头数 = num_key_value_heads / tp
#   D  ：head_dim
#   I  ：intermediate_size（MLP中间层维度）
#   V  ：vocab_size
#   tp ：张量并行数
# 以Qwen3-0.6B为例（tp=1）：H=1024，Hq=16，Hkv=8，D=128，I=3072，V=151936，28层
#
# 整个模型的结构与shape变化（Qwen3-0.6B，N个token）：
#
#   input_ids [N]
#       │ embed_tokens（查表）
#       ▼
#   x [N, 1024]
#       │
#   ════╪═══════ 以下为一个 Qwen3DecoderLayer，重复28次 ═══════
#       │
#       ├──────────────────────────────────┐ 保存一份作为残差 r
#       ▼                                  │
#   input_layernorm（RMSNorm）             │
#       ▼                                  │
#   Qwen3Attention：                       │
#     qkv_proj  [N, 1024] -> [N, 4096]     │
#     split     -> q [N, 2048]             │
#                  k [N, 1024]             │
#                  v [N, 1024]             │
#     view      -> q [N, 16, 128]          │
#                  k [N, 8, 128]           │
#                  v [N, 8, 128]           │
#     q_norm、k_norm（形状不变）           │
#     rotary_emb（形状不变）               │
#     attn      -> o [N, 16, 128]          │
#     flatten   -> [N, 2048]               │
#     o_proj    -> [N, 1024]               │
#       ▼                                  │
#      (+) <───────────────────────────────┘ x = 注意力输出 + r
#       │
#       ├──────────────────────────────────┐ 保存一份作为残差 r
#       ▼                                  │
#   post_attention_layernorm               │
#       ▼                                  │
#   Qwen3MLP：                             │
#     gate_up_proj [N, 1024] -> [N, 6144]  │
#     SiluAndMul   -> [N, 3072]            │
#     down_proj    -> [N, 1024]            │
#       ▼                                  │
#      (+) <───────────────────────────────┘ x = MLP输出 + r
#       │                                    （代码中这一步推迟到下一层的
#       │                                      input_layernorm里与归一化一起做）
#   ════╪═══════════════════════════════════════════════════
#       ▼
#   norm（最终RMSNorm）
#       ▼
#   hidden [N, 1024]
#       │ lm_head（prefill时只取每条序列最后一个token）
#       ▼
#   logits [B, 151936]


# Qwen3的自注意力模块：QKV投影 -> QK-Norm -> RoPE -> 注意力 -> 输出投影
# 张量并行：按注意力头切分，qkv_proj列并行、o_proj行并行，每层只在o_proj后做一次all_reduce
# 输入 hidden_states [N, H] -> 输出 [N, H]
class Qwen3Attention(nn.Module):

    # hidden_size：隐藏层维度H
    # num_heads：q头总数（切分前）
    # num_kv_heads：kv头总数（切分前），小于num_heads时为GQA
    # max_position：RoPE预计算cos/sin的最大位置数
    # head_dim：每个头的维度D，为None时取 hidden_size // num_heads
    # rms_norm_eps：QK-Norm的eps
    # qkv_bias：qkv投影是否带bias（Qwen2为True，Qwen3为False）
    # rope_theta：RoPE的base
    # rope_scaling：RoPE扩展配置（新版transformers中也可能包含rope_theta）
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        # total_*：模型总头数；不带total的：当前rank分到的头数
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        # Qwen3显式指定head_dim，不一定等于hidden_size / num_heads
        # 例如Qwen3-0.6B：hidden_size=1024，16个头，head_dim=128（16×128=2048≠1024）
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 本rank的q、kv在qkv_proj输出中所占的维度，用于split
        # 例如Qwen3-0.6B（tp=1）：q_size=16×128=2048，kv_size=8×128=1024
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放系数 1/sqrt(D)，防止点积过大导致softmax饱和
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # q、k、v三个投影合并为一个矩阵乘法
        # [N, H] -> [N, q_size + 2*kv_size] = [N, (Hq + 2*Hkv) * D]
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影，行并行，内部all_reduce
        # [N, Hq*D] -> [N, H]
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 新版transformers把rope_theta放在rope_scaling（rope_parameters）字典里，优先从中读取
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # QK-Norm：Qwen3新增，对每个头的q、k分别做RMSNorm（在D维上归一化，weight: [D]），稳定注意力分数
        # Qwen3去掉了qkv的bias，改用QK-Norm；Qwen2有bias、没有QK-Norm
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    # positions：[N]，每个token在其序列中的位置
    # hidden_states：[N, H]
    # 返回：[N, H]
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # [N, H] -> [N, (Hq + 2*Hkv) * D]
        qkv = self.qkv_proj(hidden_states)
        # 沿最后一维切分：q [N, Hq*D]，k [N, Hkv*D]，v [N, Hkv*D]（都是qkv的视图，不拷贝）
        # tensor.split([a, b, c], dim)：沿dim维按指定长度切成多段（chunk是平均切，split可以不等长）
        #   例如 [N, 4096].split([2048, 1024, 1024], -1) -> [N, 2048]、[N, 1024]、[N, 1024]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 拆成多头
        # tensor.view(新形状)：不改变数据，只改变"怎么解读"这些数据，元素总数必须不变
        #   -1 表示"这一维自动推算"：[N, 2048].view(-1, 16, 128) -> 2048 = 16*128，所以-1 = N
        #   例如一个token的q有2048个数，看作16个头、每头128个数
        #   [q0 q1 ... q127 | q128 ... q255 | ... ]  ->  头0: q0~q127，头1: q128~q255，...
        q = q.view(-1, self.num_heads, self.head_dim)     # [N, Hq, D]
        k = k.view(-1, self.num_kv_heads, self.head_dim)  # [N, Hkv, D]
        v = v.view(-1, self.num_kv_heads, self.head_dim)  # [N, Hkv, D]
        if not self.qkv_bias:
            # 逐头归一化，形状不变
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 加入位置信息，形状不变：q [N, Hq, D]，k [N, Hkv, D]
        q, k = self.rotary_emb(positions, q, k)
        # 注意力：o [N, Hq, D]（decode时为[B, 1, Hq, D]，N=B）
        o = self.attn(q, k, v)
        # 合并多头：flatten(1, -1) -> [N, Hq*D]；o_proj -> [N, H]
        # tensor.flatten(start, end)：把第start到第end维合并成一维，是view拆头的逆操作
        #   [N, 16, 128].flatten(1, -1) -> [N, 2048]
        #   decode时o为[B, 1, 16, 128]，flatten(1, -1) -> [B, 1*16*128] = [B, 2048]，结果一致
        output = self.o_proj(o.flatten(1, -1))
        return output


# 前馈网络（SwiGLU）：down(silu(gate(x)) * up(x))
# gate_proj和up_proj合并成一个列并行层，down_proj行并行，每层MLP只需一次all_reduce
# 输入 [N, H] -> 输出 [N, H]
class Qwen3MLP(nn.Module):

    # hidden_size：H；intermediate_size：I；hidden_act：激活函数，只支持silu
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # [N, H] -> [N, 2*I/tp]
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # [N, I/tp] -> [N, H]（内部all_reduce）
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # [N, H] -> [N, 2*I/tp]
        gate_up = self.gate_up_proj(x)
        # [N, 2*I/tp] -> [N, I/tp]
        x = self.act_fn(gate_up)
        # [N, I/tp] -> [N, H]
        x = self.down_proj(x)
        return x


# 一个Transformer解码层，Pre-Norm结构：
#   h = x + Attn(Norm(x))
#   out = h + MLP(Norm(h))
# 实现上把"残差相加"推迟到下一个Norm里与归一化融合（见RMSNorm.add_rms_forward），
# 所以每层返回(本层MLP输出, 残差)，残差相加由下一层（或最终的norm）完成
# 输入/输出的hidden_states和residual都是 [N, H]
class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            # getattr(对象, "名字", 默认值)：取属性，不存在时返回默认值（不同版本/模型的config字段可能不全）
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 注意力前的Norm，weight: [H]
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP前的Norm，weight: [H]
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # positions: [N]；hidden_states: [N, H]；residual: [N, H] 或 None（第一层）
    # 返回 (hidden_states [N, H], residual [N, H])
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层：没有上一层的残差，embedding输出本身就是残差
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：先把上一层MLP的输出加到残差上，再归一化
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # [N, H] -> [N, H]
        hidden_states = self.self_attn(positions, hidden_states)
        # 把注意力输出加到残差上，再归一化作为MLP的输入
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # [N, H] -> [N, H]
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Qwen3主干：Embedding -> N个解码层 -> 最终Norm，输出hidden_states（不含lm_head）
# input_ids [N] -> hidden_states [N, H]
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入：[N] -> [N, H]，本地权重[V/tp, H]
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # num_hidden_layers个解码层
        # nn.ModuleList：存放子模块的列表；必须用它而不是普通list，否则子模块的参数不会被登记
        # 参数名会带上下标，如 "model.layers.0.self_attn.qkv_proj.weight"，与HF权重名一致
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终归一化，weight: [H]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # [N] -> [N, H]
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            # 每层：[N, H] -> [N, H]，同时传递残差[N, H]
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层MLP输出加上残差，再做最终归一化；返回的残差不再需要
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# 因果语言模型：主干 + lm_head
# forward只算hidden_states [N, H]，compute_logits单独调用得到logits [B, V]，
# 这样CUDA Graph只捕获主干，lm_head（含跨rank的gather）放在graph外执行
class Qwen3ForCausalLM(nn.Module):
    # HF权重名片段 -> (合并后的参数名片段, 分片id)，供load_model把独立的q/k/v、gate/up权重写入合并参数
    # 例如 "...self_attn.k_proj.weight" -> 写入 "...self_attn.qkv_proj.weight" 的k段
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        # 参数名以"model."开头，与HF权重名一致，便于直接按名字加载
        self.model = Qwen3Model(config)
        # 本地权重[V/tp, H]
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 词嵌入与输出层共享权重（小模型常用，如Qwen3-0.6B），可节省一个 V×H 的矩阵
        # 共享时checkpoint里没有lm_head.weight，加载embed_tokens.weight后lm_head自动可用
        # .data = 另一个张量的.data：让两个参数指向同一块显存，修改一个另一个也变
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # input_ids [N]，positions [N] -> hidden_states [N, H]
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    # hidden_states [N, H] -> logits [B, V]（仅rank 0，其他rank为None）
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
