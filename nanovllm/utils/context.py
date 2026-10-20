from dataclasses import dataclass
import torch


# 一次前向所需的注意力元数据
# 模型的forward签名只有(input_ids, positions)，与HF保持一致；
# 注意力层需要的额外信息（序列边界、KV cache槽位、block表等）通过这个全局对象隐式传递，避免层层传参
@dataclass(slots=True)
class Context:
    # 当前是prefill还是decode，决定Attention和LMHead的计算方式
    is_prefill: bool = False
    # ---- prefill专用 ----
    # q/k的累积序列长度，用于flash_attn_varlen_func区分拼接在一起的多条序列
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # batch中q/k的最大长度，flash attention用来划分计算任务
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # ---- 通用 ----
    # 每个输入token的KV写入KV cache的位置（-1表示不写）
    slot_mapping: torch.Tensor | None = None
    # ---- decode专用 ----
    # 每条序列的上下文长度
    context_lens: torch.Tensor | None = None
    # 每条序列的block表，decode时总是需要；prefill仅在有缓存前缀时需要
    block_tables: torch.Tensor | None = None

# 模块级全局变量，同一进程内所有层共享
_CONTEXT = Context()

def get_context():
    return _CONTEXT

# 每次前向之前由ModelRunner设置
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

# 前向结束后重置，释放对张量的引用
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
