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


# Qwen3的自注意力模块：QKV投影 -> QK-Norm -> RoPE -> 注意力 -> 输出投影
# 张量并行：按注意力头切分，qkv_proj列并行、o_proj行并行，每层只在o_proj后做一次all_reduce
class Qwen3Attention(nn.Module):

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
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放系数 1/sqrt(head_dim)，防止点积过大导致softmax饱和
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # q、k、v三个投影合并为一个矩阵乘法
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影，行并行，内部all_reduce
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
        # QK-Norm：Qwen3新增，对每个头的q、k分别做RMSNorm，稳定注意力分数，训练更稳定
        # Qwen3去掉了qkv的bias，改用QK-Norm；Qwen2有bias、没有QK-Norm
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    # positions：[N]，每个token在其序列中的位置
    # hidden_states：[N, hidden_size]，batch内所有token拼接成一维（没有batch维）
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 拆成多头：[N, heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # 合并多头：[N, heads, head_dim] -> [N, heads*head_dim]
        output = self.o_proj(o.flatten(1, -1))
        return output


# 前馈网络（SwiGLU）：down(silu(gate(x)) * up(x))
# gate_proj和up_proj合并成一个列并行层，down_proj行并行，每层MLP只需一次all_reduce
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


# 一个Transformer解码层，Pre-Norm结构：
#   h = x + Attn(Norm(x))
#   out = h + MLP(Norm(h))
# 实现上把"残差相加"推迟到下一个Norm里与归一化融合（见RMSNorm.add_rms_forward），
# 所以每层返回(本层输出, 残差)，残差相加由下一层（或最终的norm）完成
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
        # 注意力前的Norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP前的Norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.self_attn(positions, hidden_states)
        # 把注意力输出加到残差上，再归一化作为MLP的输入
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Qwen3主干：Embedding -> N个解码层 -> 最终Norm，输出hidden_states（不含lm_head）
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层MLP输出加上残差，再做最终归一化；返回的残差不再需要
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# 因果语言模型：主干 + lm_head
# forward只算hidden_states，compute_logits单独调用，
# 这样CUDA Graph只捕获主干，lm_head（含跨rank的gather）放在graph外执行
class Qwen3ForCausalLM(nn.Module):
    # HF权重名 -> (合并后的参数名, 分片id)，供load_model把独立的q/k/v、gate/up权重写入合并参数
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
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 词嵌入与输出层共享权重（小模型常用，如Qwen3-0.6B），可节省一个 vocab×hidden 的矩阵
        # 共享时checkpoint里没有lm_head.weight，加载embed_tokens.weight后lm_head自动可用
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
