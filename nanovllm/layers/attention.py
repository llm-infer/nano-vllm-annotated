import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# Shape记号
#   N：本次前向的token总数（prefill为拼接后的总长度，decode为batch大小）
#   B：序列数
#   Hq/Hkv：当前rank计算的q/kv的头数
#   D：头的维度，head_dim
#   NB：KV cache的block数量
#   BS：每个KV block容纳多少token
#   M：block_table的列数
#
# 基础概念：
#   1. 不管张量几维，在显存中都是一段连续的一维数组，形状就是解读张量的工具
#   2. 步长stride(i)：第i维的下标+1时，在一维数组中需要前进多少步
#       例如形状[2, 3]的连续张量[[a, b, c], [d, e, f]]在内存中为[a b c d e f]
#       stride(0) = 3，a跳到d，需要前进3步，stride(1) = 1，a跳到b需要前进1步
#       元素[i][j]的位置 = i * stride(0) + j * stride(1)
#       切片得到的视图共用原内存，stride可能不连续。例如qkv形状[N, 4096]，k = qkv[:, 2048:3072] 形状[N, 1024]，但stride(0)仍是4096（跳过整行）
#
# attention的数据流
#   q [N, Hq, D]     k [N, Hkv, D]     v [N, Hkv, D]
#        │                  │               │
#        │                  └──────┬────────┘
#        │                         ▼
#        │              store_kvcache：按slot_mapping把本地的k/v写入KV cache
#        │                         │
#        ▼                         ▼
#    flash_attn：从KV cache（decode）或本次的k/v（prefill）中读取kv，与q计算注意力
#        │
#        ▼
#   o [N, Hq, D]       V 按注意力权重加权求和的结果
#        │
#        ▼
#   o_proj（行并行） -> [N, hidden] 各卡部分和，all_reduce 汇总
#                                 此时的 [N, hidden] 与x形状相同，本质上是感知了上下文语义的x


# Triton：用python语法写GPU kernel的语言，@triton.jit装饰的函数会编译成GPU代码
#   kernel：在GPU上执行的代码；启动时需要指定grid（启动多少program），每个program执行相同代码
#   通过 tl.program_id(0) 得知自己是第几个program，以便处理自己负责的数据
#   xxx_ptr是显存的地址，ptr+offset得到一组地址，通过tl.load/tl.store批量读写
#
# Triton kernel：把本地计算的k/v写入KV cache指定槽位
# 每个program处理一个token（grid=（N，）），把该token所有K、V（共Hkv*head_dim个元素）拷贝到cache中
# key_ptr/value_ptr：本次新算出的KV（即将要写入kv cache的 kv），逻辑形状[N, Hkv, D]
# key_stride/value_stride：第0维（token维度）的步长，k/v是qkv切出的视图，步长可能大于D
# k_cache_ptr/v_cache_ptr：本层的kv cache，逻辑形状[NB*BS, Hkv*D]
# slot_mappint：[N]，本层每个token的kv要写入的物理位置，slot = block_id * block_size + offset
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    # D=Hkv*head_dim
    # 固定的数，用triton编译器常量。triton会为不同的编译器分别编译kernel
    # 好处：让形状静态化，编译器分配好寄存器，和批量访存；消除运行时分支；循环展开；常量折叠等等
    D: tl.constexpr,
):
    # 当前program处理第idx个token
    idx = tl.program_id(0)
    # 找到目标token的kv写入kv cache的物理地址
    slot = tl.load(slot_mapping_ptr + idx)
    # -1表示不需要写（CUDA Graph中padding的位置，bs=15时，会补全1个位置到16，然后用bs=16的cuda graph）
    if slot == -1: return
    # 源地址：第idx个token的起始位置 + [0, D), 一次读取一个长度为D的向量
    # 例如：idx=2，key_stride=4096，D=1024，读取位置[8192, 8193, ..., 9215]
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 从源地址中读出k/v，长度为D
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # cache 按[NB*BS, D]，一个token的k/v占用一个槽位，第slot个槽位起始于slot*D
    #   元素下标：  0 1 2 3 | 4 5 6 7 | 8 9 10 11 | ...
    #               slot 0   slot 1    slot 2
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# 工作流程（A计算6个token，block_table=[4, 1]；B计算3个token，block_table=[5]；N=9、BS=4）
#   key [9, Hkv, head_dim]      启动9个program      k_cache展平成 [NB*BS行，Hkv*head_dim列]
#   每行是一个token的所有kv     每个program处理一行数据         每行是1个槽位
#   行0  a0   ────────────────── program 0  ─────────> 第16行（block 4 偏移0）
#   行1  a1   ────────────────── program 1  ─────────> 第17行（block 4 偏移1）
#   ...                          ...                 ...
#   行4  a4   ────────────────── program 4  ─────────> 第4行（block 1 偏移0）
#   行5  a5   ────────────────── program 5  ─────────> 第5行（block 1 偏移1）
#   行6  b4   ────────────────── program 6  ─────────> 第20行（block 5 偏移0）
#   ...                          ...                 ...
#   行8  b6   ────────────────── program 8  ─────────> 第22行（block 5 偏移2）
#
#   每个program只做一件事，读slot_mapping[自己的编号]，把key的这一行（Hkv*head_dim个数字）拷贝到k_cache的第slot行
#   slot=-1时直接跳过（CUDA graph凑batch加的padding行）
#   B的b0~b3不再本次计算的kv里，它们的kv早就已经写入到kv_cache里（来自请求留下的缓存，或者上一个chunked prefill）
#
# key/value：[N, Hkv, head_dim]
# k_cache/v_cache：[NB, BS, Hkv, head_dim]
# slot_mapping：[N]
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    # 每个K/V的元素个数
    D = num_heads * head_dim
    # 内存布局检查，kv后两维必须是连续内存
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # kcache/v_cache前进一个BS，需要跳过Hkv*head_dim个元素
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动N个program，每个program处理一个token
    # kernel[grid](args...)是trition的启动语法；key/value张量会被自动转换为显存地址传入
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# 注意力计算层（不含QKV投影和输出投影），负责写KV cache和调用flash attn
#   输入：q、k、v，来自qkv_proj(x)
#   输出：o，可以理解为"看完上下文之后"得到的信息摘要
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        # q头数，Hq
        self.num_heads = num_heads
        # 每个头的维度
        self.head_dim = head_dim
        # softmax前对QK^T的缩放系数，一般为1/sqrt(D)
        self.scale = scale
        # 当前rank的kv头数，Hkv
        self.num_kv_heads = num_kv_heads
        # 占位，ModelRunner.allocate_kv_cache会替换为本层的cache视图，[NB, BS, Hkv, head_dim]
        self.k_cache = self.v_cache = torch.tensor([])

    # forward的三种情况
    #     情况                     本次k/v写入cache？      K/V从哪里读                 调用的函数
    #     ─────────────────────────────────────────────────────────────────────────────────────────────────────
    #     warmup（cache未分配）     跳过                  本次计算的 k/v        flash_attn_varlen_func
    #     prefill，无缓存前缀       写入                  本次计算的 k/v        flash_attn_varlen_func
    #     prefill，有缓存前缀       写入               kv cache，本次计算的k/v   flash_attn_varlen_func，传block_table
    #     decode                  写入                    kv cache          flash_attn_with_kvcache
    #
    # 输入：q [N, Hq, head_dim], k/v [N, Hkv, head_dim]
    # 输出：o [N, Hq, head_dim]
    # Hq > Hkv时为GQA（每Hq/Hkv个q头共享一组kv），flash attn内部自动处理
    #   例如Hq=16，Hkv=8：
    #     q头 0,1 ──> kv头0
    #     q头 2,3 ──> kv头1
    #     ...
    #     q头14,15──> kv头7
    #   KV cache只需存8个头，显存减半
    # 
    # 注意力计算，每个头独立，互相无依赖：o = softmax(q @ k^T * scale + 因果掩码) @ v
    #   q @ K^T：每个q与每个k的相似度，prefill时为 [seq_len, seq_len], decode时为[1, seq_len]
    #   因果掩码：token只能看到自己和之前的token，未来位置的分数置为-∞，softmax后为0
    #           prefill必须用，为了和训练期间保持一致，decode天然满足，不需要掩码
    #   因果掩码示意（4个token，✓可见，✗不可见）：
    #            key: t0 t1 t2 t3
    #     query t0:   ✓  ✗  ✗  ✗
    #     query t1:   ✓  ✓  ✗  ✗
    #     query t2:   ✓  ✓  ✓  ✗
    #     query t3:   ✓  ✓  ✓  ✓
    #
    # flash attn：把上面的计算融合成一个kernel、分块计算、不在显存中保存完整的注意力矩阵
    # 显存占用从O(N^2)降到O(N)，速度也更快
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 先把本次计算好的k/v存到kv_cache里
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # 有缓存时，由于本次计算的k/v只是后半段，需要连同前缀一起参与计算
            # 由于本次计算的k/v已经写入cache，k_cache/v_cache就是完整的，直接赋值给k、v
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 变长注意力：多条序列拼接在一起，按cu_seqlens区分边界，无需padding
            # causal=True且q比k短时，flash attention按右下角对齐因果掩码，
            #               c0  c1  t2  t3
            #       q(t2)    ✓   ✓   ✓   ✗
            #       q(t3)    ✓   ✓   ✓   ✓
            # 
            # 整个batch的可见关系
            #                    A  k                   │            B  k
            #            a0 a1 a2 a3 a4 a5              │ b0 b1 b2 b3 b4 b5 b6
            #            ┌──────────────────────────────┼──────────────────────────
            #      a0    │ ✓  ·  ·  ·  ·  ·             │ ·  ·  ·  ·  ·  ·  ·
            #      a1    │ ✓  ✓  ·  ·  ·  ·             │ ·  ·  ·  ·  ·  ·  ·
            #    A a2    │ ✓  ✓  ✓  ·  ·  ·             │ ·  ·  ·  ·  ·  ·  ·
            #    q a3    │ ✓  ✓  ✓  ✓  ·  ·             │ ·  ·  ·  ·  ·  ·  ·
            #      a4    │ ✓  ✓  ✓  ✓  ✓  ·             │ ·  ·  ·  ·  ·  ·  ·
            #      a5    │ ✓  ✓  ✓  ✓  ✓  ✓             │ ·  ·  ·  ·  ·  ·  ·
            #            ├──────────────────────────────┼──────────────────────────
            #    B b4    │ ·  ·  ·  ·  ·  ·             │ ✓  ✓  ✓  ✓  ✓  ·  ·
            #    q b5    │ ·  ·  ·  ·  ·  ·             │ ✓  ✓  ✓  ✓  ✓  ✓  ·
            #      b6    │ ·  ·  ·  ·  ·  ·             │ ✓  ✓  ✓  ✓  ✓  ✓  ✓
            #            └──────────────────────────────┴──────────────────────────
            # 不同序列之间互相看不见（由 cu_seqlens 划分边界），每条序列只和自己的k做因果注意力
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # 每条序列只有一个q，q [B, Hq, D] -> unsqueeze(1) -> [B, 1, Hq, D]，flash attn要求q是4维 [batch, seqlen, heads, D]
            # k_cache/v_cache：[NB, BS, Hkv, D]
            # cache_seqlens: [B], block_table: [B, M]
            # o [B, 1, Hq, D]
            # 通过block_table从分页的KV cache中读取前ccache_seqlens个位置的kv
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        # prefill: [N, Hq, D]；decode: [B, 1, Hq, D]
        # 调用方用flatten(1, -1)展平为 [N, Hq*D]（decode时 [B, 1*Hq*D]），两种情况结果一致
        return o
