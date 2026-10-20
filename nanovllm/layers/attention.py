import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# Triton kernel：把本次计算出的K/V写入KV cache的指定槽位
# 每个program处理一个token（grid = (N,)），把该token所有kv头的K、V（共D个元素）拷贝到cache中
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    # tl.constexpr：编译期常量，Triton会为不同的D分别编译，tl.arange的长度必须是编译期常量
    D: tl.constexpr,
):
    # 当前program处理第idx个token
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    # -1表示不需要写入（CUDA Graph中padding出来的位置）
    if slot == -1: return
    # 源地址：第idx个token的起始位置 + [0, D)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 目标地址：cache按[num_blocks * block_size, D]连续排布，第slot个槽位起始于slot * D
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# key/value: [N, num_kv_heads, head_dim]，N为本次计算的token数
# k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 内存布局检查：kernel把每个token的所有头当作长度为D的连续一段来读写
    # key/value的后两维必须是连续的（它们是qkv split出来的视图，第0维stride可能大于D）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # cache中相邻槽位间隔正好D
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# 注意力计算层（不含QKV投影和输出投影），负责写KV cache并调用flash attention
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 占位的空张量，ModelRunner.allocate_kv_cache会替换为本层的cache视图
        self.k_cache = self.v_cache = torch.tensor([])

    # q: [N, num_heads, head_dim]，k/v: [N, num_kv_heads, head_dim]
    # num_heads > num_kv_heads时为GQA（多个q头共享一组kv头），flash attention内部自动处理
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 先把本次的K/V写入cache（预热阶段cache还未分配，跳过）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # 有缓存前缀时，本次的k/v只是后半段，需要连同前缀一起参与注意力
            # 由于新K/V已写入cache，直接把整个cache和block_table交给flash attention按页读取完整的K/V
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 变长注意力：多条序列拼接在一起，按cu_seqlens区分边界，无需padding
            # causal=True且q比k短时，flash attention按右下角对齐因果掩码，
            # 即q的第i个token能看到k中前 (seqlen_k - seqlen_q + i + 1) 个token，正好是"缓存前缀+自身之前"
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # 每条序列只有1个query token，q.unsqueeze(1)变成[batch, 1, num_heads, head_dim]
            # 通过block_table从分页的KV cache中读取每条序列前context_lens个位置的K/V（PagedAttention）
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
