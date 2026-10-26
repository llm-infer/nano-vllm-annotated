from dataclasses import dataclass
import torch


# 一次前向所需的注意力元数据
# 模型的forward函数参数只有input_ids、positions
# 注意力层需要的额外信息（序列边界、kv cache槽位、block表等等）通过Context传递，避免层层传参
@dataclass(slots=True)
class Context:
    # prefill/decode用不同的flash attn接口
    is_prefill: bool = False
    # ----prefill专用----
    # 第i条序列本次计算的token在拼接张量中的区间为[cu_seqlens_q[i], cu_seqlens_q[i+1])
    # 也被LMHead用来取每条序列最后一个token：cu_seqlens_q[1:]-1
    cu_seqlens_q: torch.Tensor | None = None
    # k包含已缓存的前缀
    # 例如seq缓存256个，本次计算100个 -> q长100，k长356
    cu_seqlens_k: torch.Tensor | None = None
    # batch中单条序列q/k长度最大值，flash attention用来规划计算任务
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # ----通用----
    # 每个token的KV写入kv_cache的槽位，[N]
    # slot = 物理block_id × block_size + block内偏移；-1表示不写
    slot_mapping: torch.Tensor | None = None
    # ----decode----
    # 每条序列的上下文长度，[B]
    # 注意力模块中，读取前context_lens个kv cache
    context_lens: torch.Tensor | None = None
    # 每条序列的block_table，[B, M]
    # block_tables[i][j]：第i条序列第j个block对应的物理block id
    # decode需要；prefill仅在有前缀缓存是需要
    block_tables: torch.Tensor | None = None

# 模块级全局变量，同一进程内所有层共享
# 变量名以下划线开头，约定模块内部使用，外部应通过get_context/set_context访问
# 使用方法：
# ModelRunner.run()
#     │ set_context(...)
#     │ model(input_ids, positions)
#     │    └─ 每一层 Attention.forward -> get_context()读取slot_mapping、cu_seqlens
#     │    └─ ParallelLMHead.forward  -> get_context()读取is_prefill, cu_seqlens_q
#     │ reset_context()
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
