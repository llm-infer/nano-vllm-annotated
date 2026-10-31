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


# shape记号
#   N：本次前向计算的token数（batch内所有序列拼成一维，没有batch维）
#   B：序列数
#   H：hidden_size
#  Hq：当前rank负责计算的q头数 = num_attention_heads/tp
# Hkv：当前rank负责计算的k/v头数 = num_key_value_heads/tp
#   D：head_dim
#   I：intermediate_size（MLP中间层维度）
#   V：vocab_size
#  tp：张量并行数
#
# 以Qwen3-0.6B为例：H=1024，Hq=16，Hkv=8，D=128，I=3072，V=151936,28层
# 
# 整个模型的结构和shape变化
#   input_ids [N]
#       │ embed_tokens(查表)
#       ▼
#   x [N, 1024]
#       │
#   ════╪═══════ 以下为一个Qwen3DecoderLayer，重复28次 ═══════
#       │
#       ├──────────────────────────────────┐ 保存一份作为残差 r
#       ▼                                  │
#   input_layernorm（RMSNorm）              │
#       ▼                                   │
#   Qwen3Attention：                        │
#       qkv_proj [N, 1024] -> [N, 4096]     │
#       split       ->  q [N, 2048]         │
#                       k [N, 1024]         │
#                       v [N, 1024]         │
#       view        ->  q [N, 16, 128]      │
#                       k [N, 8   128]      │
#                       v [N, 8   128]      │
#       q_norm、k_norm（形状不变）             │
#       rotary_emb(形状不变)                  │
#       attn        ->  o [N, 16, 128]       │
#       flatten     ->  [N, 2048]            │
#       o_proj      ->  [N, 1024]            │
#       ▼                                    │
#      (+) <─────────────────────────────────┘ x = 注意力输出 + r
#       │
#       ├────────────────────────────────────┐ 保存一份作为残差 r
#       ▼                                    │
#   post_attention_layernorm                 │
#       ▼                                    │  
#   Qwen3MLP                                 │
#       gate_up_proj [N, 1024] -> [N, 6144]  │
#       SiluAndMul  -> [N, 3072]             │
#       down_proj   -> [N, 1024]             │
#       ▼                                    │
#      (+) <─────────────────────────────────┘ x = MLP输出 + r
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
# 张量并行：按注意力头切分，qkv_proj列并行，o_proj行并行，每层只在o_proj后做一次all_reduce
# 输入 [N, H] -> 输出 [N, H] (带上下文信息)
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        # q头数（切分前）
        num_heads: int,
        # kv头数（切分前）
        num_kv_heads: int,
        # RoPE预计算cos/sin最大位置数
        max_position: int = 4096 * 32,
        # 每个头的维度，为None时取hidden_size//num_heads
        head_dim: int | None = None,
        # QK-Norm的eps，防止除0
        rms_norm_eps: float = 1e-06,
        # qkv投影是否带bias
        qkv_bias: bool = False,
        # RoPE的base
        rope_theta: float = 10000,
        # RoPE拓展配置
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 本rank的一个q、k、v所占的维度，用于split
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放系数，防止点积过大导致softmax饱和
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # q、k、v三个投影合并成一个矩阵乘法
        # [N, H] -> [N, q_size + 2*kv_size] = [N, (Hq + 2*Hkv) * D]
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影，行并行，内部all_reduce
        # [N, Hq*D] -> [N， H]
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
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
        # QK-Norm：对q、k每个头做RMSNorm，在D维上做归一化，weight [D]，稳定数值计算
        # Qwen3去掉了qkv_bias，改用QK-Norm
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    # positions：[N]，每个token在其序列中的位置
    # hidden_state：[N, H]
    # 返回：[N, H]
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # [N, H] -> [N, (Hq + 2*Hkv) * D]
        qkv = self.qkv_proj(hidden_states)
        # 沿最后一维切分
        # q: [N, Hq*D]
        # k/v: [N, Hkv*D]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 拆成多头
        # q: [N, Hq*D] -> [N, Hq, D]
        # k/v: [N, Hkv*D] -> [N, Hkv, D]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # qk-norm，形状不变
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 加入位置信息，形状不变
        q, k = self.rotary_emb(positions, q, k)
        # 计算注意力
        # o [N, Hq, D]
        o = self.attn(q, k, v)
        # 合并多头
        # o.flatten(1, -1): 把(start, end)合并成一维，[N, Hq, D] -> [N, Hq * D]
        # o_proj: [N, Hq * D] -> [N, H]
        output = self.o_proj(o.flatten(1, -1))
        return output


# 前馈网络 SwiGLU：down(silu(gate(x))*up(x))
# gate_proj和up_proj合并成一个列并行层，down_proj行并行，每层MLP只需要一次all_reduce
# 输入 [N, H] -> 输出 [N, H]
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        # 激活函数，支持silu
        hidden_act: str,
    ) -> None:
        super().__init__()
        # [N, H] -> [N, I*2/tp]
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # [N, I/tp] -> [N, H/tp] -> [N, H] 内部all_reduce
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


# Transformer解码层，包含pre-norm、attn、post-norm、mlp模块
#   h = x + attn(norm(x))
#   out = h + mlp(norm(h))
# 代码实现把每层最后一次“残差相加”放到了下一层的norm里（见RMSNOrm.add_rms_forward）
# 输入/输出的hidden_states和残差redidual都是[N, H],直接逐元素相加
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
        # 注意力前的norm，weight [H]
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # mlp前的norm，weight [H]
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # hidden_state、residual在层与层之间如何传递（e=embedding输出，a_i/m_i是第1层注意力/MLP输出
    # 第0层：redisual = None
    #   input_layernorm(e)              -> 输入 Norm(e)         residual = e
    #   self_attn                       -> a0
    #   post_attention_layernorm(a0, e) -> 输入 Norm(e + a0)    residual = e + a0
    #   mlp                             -> m0
    #   返回 (m0, e+a0)                 注意：m0还没有加到残差上
    #
    # 第1层：residual 不是 None
    #   input_layernorm(m0, e+a0)       -> 输入 Norm(e+a0+m0)   residual = e + a0 + m0
    #                                                            ↑ 在这里补上第0层的加法
    #   ...
    #
    # 第27层 (m27, residual)
    #   norm(m27, residual)             -> 补上最后一次加法并归一化，得到最终hidden_state
    #  
    # 入参：positons [N], hidden_states [N, H], residual [N, H]或None
    # 返回：hidden_states [N, H], residual [N, H]
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层：没有上一层的残差，embedding本身就是残差
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：先把上一层MLP的输出加到残差上，再归一化
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Qwen3主干：embedding -> N个decoder -> 最终norm，输出hidden_states（不含lm_head）
# input_ids [N] -> hidden_state [N, H]
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # embedding [N] -> [N, H]
        # rank本地权重 [V/tp, H]
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # num_hidden_layers个解码层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终的归一化，weight [H]
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
            # 每层 [N, H] -> [N, H]，同时传递残差 [N, H]
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层的mlp输出加上残差，再做归一化，返回的残差不再需要
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# 模型入口：主干+lm_head
# forward只算hidden_states，compute_logits单独调用得到logits [B, V]
# 这样CUDA graph只捕获forward，lm_head放在graph外执行
# lm_head不适合放到CUDA graph的理由
#   输出不一致：跨rank的gather，rank0输出[B, V]，其他rank输出None
class Qwen3ForCausalLM(nn.Module):
    # 供load_model把q/k/v, gate/up权重合并
    # 例如..self_attn.k_proj.weight -> self_attn.qkv_proj的k段
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
        # 命名必须是model，与HF权重名一致，便于直接按名字加载
        self.model = Qwen3Model(config)
        # 本地权重 [V/tp, H]
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # embedding与输出层共享权重，可节省一个V*H的矩阵
        if config.tie_word_embeddings:
            # 让两个参数指向同一块显存，修改一个另一个也变
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # input_ids [N], positions [N] -> hidden_states [N, H]
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    # hidden_states [N, H] -> logits [B, V]，仅rank0，其他rank 为none
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
