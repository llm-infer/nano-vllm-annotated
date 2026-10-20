from dataclasses import dataclass
import torch


# 一次前向所需的注意力元数据
# 模型的forward签名只有(input_ids, positions)，与HF保持一致；
# 注意力层需要的额外信息（序列边界、KV cache槽位、block表等）通过这个全局对象隐式传递，避免层层传参
# shape记号：B 序列数，N 本次计算的token总数，M block_table列数
@dataclass(slots=True)
class Context:
    # 当前是prefill还是decode，决定Attention用哪个flash attention接口、LMHead是否只取最后一个token
    is_prefill: bool = False
    # ---- prefill专用 ----
    # q的累积序列长度，[B+1]，int32
    # 第i条序列本次计算的token在拼接张量中的区间为[cu_seqlens_q[i], cu_seqlens_q[i+1])
    # 也被LMHead用来取每条序列最后一个token：cu_seqlens_q[1:] - 1
    cu_seqlens_q: torch.Tensor | None = None
    # k的累积序列长度，[B+1]，int32；k包含已缓存的前缀，所以每段长度 >= q
    # 例如：seq缓存256个、本次计算100个 -> 该序列q长100，k长356
    cu_seqlens_k: torch.Tensor | None = None
    # batch中单条序列q/k长度的最大值（Python int），flash attention用来划分计算任务
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # ---- 通用 ----
    # 每个输入token的KV写入KV cache的槽位，[N]，int32
    # slot = 物理block_id × block_size + block内偏移；-1表示不写（CUDA Graph的padding）
    slot_mapping: torch.Tensor | None = None
    # ---- decode专用 ----
    # 每条序列的上下文长度（含本次输入的token），[B]，int32
    # 注意力只读取cache中每条序列前context_lens个位置
    context_lens: torch.Tensor | None = None
    # 每条序列的block表，[B, M]，int32，不足部分填-1
    # block_tables[i][j]：第i条序列第j个逻辑block对应的物理block id
    # decode时总是需要；prefill仅在有缓存前缀时需要（否则为None）
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
