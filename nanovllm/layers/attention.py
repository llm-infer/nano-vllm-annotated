import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# shape记号：
#   N  ：本次前向的token总数（prefill为拼接后的总长度，decode为batch大小B）
#   B  ：序列数
#   Hq ：当前rank的q头数；Hkv：当前rank的kv头数
#   D  ：head_dim
#   NB ：KV cache的block数；BS：block_size；M：block_table列数
#
# 基础概念：张量在内存中的存放方式与stride（步长）
#   不管张量有几维，在显存里都是一段连续的一维数组，形状只是"怎么解读"这段数组
#   stride(i)：第i维的下标+1时，在一维数组中要跳过多少个元素
#   例如形状[2, 3]的连续张量 [[a, b, c], [d, e, f]]，内存中为 [a b c d e f]
#     stride(0) = 3（从a到d跳3个），stride(1) = 1（从a到b跳1个）
#     元素[i][j]的位置 = i * stride(0) + j * stride(1)
#   切片得到的视图共用原内存，stride可能"不连续"：
#     qkv形状[N, 4096]，k = qkv[:, 2048:3072] 形状[N, 1024]，但stride(0)仍是4096（跳过整行）
#
# 本层的数据流（一次前向、一层）：
#
#   q [N, Hq, D]   k [N, Hkv, D]   v [N, Hkv, D]
#        │              │               │
#        │              └──────┬────────┘
#        │                     ▼
#        │        store_kvcache：按slot_mapping把本次的K/V写进KV cache
#        │                     │
#        ▼                     ▼
#   flash attention（从KV cache或本次的k/v中读取K/V，与q做注意力）
#        │
#        ▼
#   o [N, Hq, D]


# Triton：用Python语法写GPU kernel的语言，@triton.jit 装饰的函数会被编译成GPU代码
#   kernel：在GPU上并行执行的函数；启动时指定grid（启动多少个"program"），每个program执行同一段代码，
#   通过 tl.program_id(0) 得知自己是第几个，从而处理不同的数据
#   这里的参数 xxx_ptr 是显存地址（指针），ptr + offsets 得到一组地址，tl.load/tl.store 批量读写
#
# Triton kernel：把本次计算出的K/V写入KV cache的指定槽位
# 每个program处理一个token（grid = (N,)），把该token所有kv头的K、V（共D个元素，这里D=Hkv*head_dim）拷贝到cache中
# key_ptr / value_ptr：本次的K/V，逻辑形状[N, Hkv, head_dim]
# key_stride / value_stride：第0维（token维）的步长（元素个数），K/V是qkv split出的视图，步长可能大于D
# k_cache_ptr / v_cache_ptr：本层KV cache，逻辑形状[NB*BS, D]
# slot_mapping_ptr：[N]，每个token写入的槽位
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
    # 源地址：第idx个token的起始位置 + [0, D)，一次读取一个长度为D的向量
    # tl.arange(0, D)：生成 [0, 1, ..., D-1]，类似Python的range，但是是一个向量
    # 例如 idx=2，key_stride=4096，D=1024：读取位置 [8192, 8193, ..., 9215]
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)        # [D]
    value = tl.load(value_ptr + value_offsets)  # [D]
    # 目标地址：cache按[NB*BS, D]连续排布，第slot个槽位起始于slot * D
    # 例如 slot=968，D=1024：写入位置 [991232, ..., 992255]
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# key/value: [N, Hkv, head_dim]
# k_cache/v_cache: [NB, BS, Hkv, head_dim]
# slot_mapping: [N]
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    # 每个token的K（或V）元素个数
    D = num_heads * head_dim
    # 内存布局检查：kernel把每个token的所有头当作长度为D的连续一段来读写
    # assert 条件：条件为False时直接报错，用来提前发现不符合预期的输入
    # key/value的后两维必须是连续的（它们是qkv split出来的视图，第0维stride可能大于D）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # cache中相邻槽位间隔正好D，即[NB, BS, Hkv, head_dim]可以看作[NB*BS, D]
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # grid=(N,)：启动N个program，每个处理一个token
    # kernel[grid](参数...) 是Triton的启动语法；key/value等张量会被自动转换成显存地址传入
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
        # 当前rank的q头数Hq
        self.num_heads = num_heads
        # 每个头的维度D
        self.head_dim = head_dim
        # softmax前对QK^T的缩放系数，一般为1/sqrt(D)
        self.scale = scale
        # 当前rank的kv头数Hkv
        self.num_kv_heads = num_kv_heads
        # 占位的空张量，ModelRunner.allocate_kv_cache会替换为本层的cache视图 [NB, BS, Hkv, D]
        self.k_cache = self.v_cache = torch.tensor([])

    # 输入：q [N, Hq, D]，k/v [N, Hkv, D]
    # 输出：o [N, Hq, D]
    # Hq > Hkv时为GQA（每 Hq/Hkv 个q头共享一组kv头），flash attention内部自动处理
    #   例如Hq=16，Hkv=8：
    #     q头 0,1 ──> kv头0
    #     q头 2,3 ──> kv头1
    #     ...
    #     q头14,15──> kv头7
    #   KV cache只需存8个头，显存减半
    #
    # 注意力计算（每个头独立）：o = softmax(q @ k^T * scale + 因果掩码) @ v
    #   q @ k^T：每个query token与每个key token的相似度
    #   因果掩码：token只能看到自己和之前的token（看不到未来），未来位置的分数设为-∞，softmax后为0
    #
    #   因果掩码示意（4个token，✓可见，✗不可见）：
    #            key: t0 t1 t2 t3
    #     query t0:   ✓  ✗  ✗  ✗
    #     query t1:   ✓  ✓  ✗  ✗
    #     query t2:   ✓  ✓  ✓  ✗
    #     query t3:   ✓  ✓  ✓  ✓
    #
    # flash attention：把上面的计算融合成一个kernel，分块计算、不在显存中保存完整的分数矩阵，
    # 显存占用从O(N²)降到O(N)，速度也更快
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 先把本次的K/V写入cache（预热阶段cache还未分配，跳过）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # 有缓存前缀时，本次的k/v只是后半段，需要连同前缀一起参与注意力
            # 由于新K/V已写入cache，直接把整个cache和block_table交给flash attention按页读取完整的K/V
            # 此时k/v从 [N, Hkv, D] 变为 [NB, BS, Hkv, D]（分页格式）
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 变长注意力：多条序列拼接在一起，按cu_seqlens区分边界，无需padding
            #   q: [N, Hq, D]，cu_seqlens_q: [B+1]
            #   k/v: [N, Hkv, D]（无前缀缓存）或 [NB, BS, Hkv, D] + block_table [B, M]（有前缀缓存）
            #   o: [N, Hq, D]
            # causal=True且q比k短时，flash attention按右下角对齐因果掩码，
            # 即q的第i个token能看到k中前 (seqlen_k - seqlen_q + i + 1) 个token，正好是"缓存前缀+自身之前"
            # 例如：缓存256个、本次计算100个 -> 本次第0个token（位置256）能看到前257个token
            #
            #   有前缀缓存时的掩码（缓存2个token c0 c1，本次计算2个token t2 t3）：
            #              key: c0 c1 t2 t3
            #     query t2:     ✓  ✓  ✓  ✗
            #     query t3:     ✓  ✓  ✓  ✓
            #   q只有2行，k有4列，掩码靠右下角对齐
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # 每条序列只有1个query token，N = B
            #   q: [B, Hq, D] -> unsqueeze(1) -> [B, 1, Hq, D]（第1维是每条序列的query长度）
            #     unsqueeze(i)：在第i个位置插入一个长度为1的新维度，数据不变，只是多了一层括号
            #     例如 [2, 3] 的张量 unsqueeze(1) 变成 [2, 1, 3]：
            #       [[a, b, c],          [[[a, b, c]],
            #        [d, e, f]]    ->     [[d, e, f]]]
            #     这里是因为flash_attn_with_kvcache要求q是4维 [batch, seqlen, heads, D]
            #   k_cache/v_cache: [NB, BS, Hkv, D]
            #   cache_seqlens: [B]，block_table: [B, M]
            #   o: [B, 1, Hq, D]
            # 通过block_table从分页的KV cache中读取每条序列前context_lens个位置的K/V（PagedAttention）
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        # prefill: [N, Hq, D]；decode: [B, 1, Hq, D]
        # 调用方用flatten(1, -1)展平为 [N, Hq*D]（decode时 [B, 1*Hq*D]），两种情况结果一致
        return o
