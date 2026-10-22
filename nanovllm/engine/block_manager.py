from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


#====================================================================================================
# 全局视角：BlockManager、block_table、CPU、显存、ModelRunner.kv_cache 之间的关系
#====================================================================================================
#
# 一句话总结：
#   BlockManager 在 CPU 上"记账"（哪个block空闲、哪个被谁用、内容的hash），只处理整数编号，从不碰显存；
#   ModelRunner.kv_cache 是 GPU 上一整块预先分配好的显存，真正存放K/V数据；
#   两者之间唯一的纽带是"物理block编号"：block编号 = kv_cache 第2维（NB维）的下标。
#
# 一、整体结构（以 tensor_parallel_size=1 为例）
#
#   ┌────────────────────────── CPU 内存（rank 0 主进程，普通 Python 对象） ───────────────────────────┐
#   │ Scheduler                                                                                        │
#   │   └─ BlockManager（只记账，不碰显存）                                                            │
#   │        blocks[i]         Block(block_id=i, ref_count, hash, token_ids)  每个物理block的元信息    │
#   │        free_block_ids    deque([2, 5, 8, ...])   空闲的block编号                                 │
#   │        used_block_ids    {3, 7, ...}             正在使用的block编号                             │
#   │        hash_to_block_id  {hash: block_id}        前缀缓存索引                                    │
#   │                                                                                                  │
#   │ Sequence（每条请求一个）                                                                         │
#   │   token_ids   = [t0, t1, ..., t299]      token本身只存在CPU上                                    │
#   │   block_table = [7, 3]                   第0个逻辑block在物理block 7，第1个在物理block 3         │
#   └──────────────────────────────────────────────────────────────────────────────────────────────────┘
#                    │ BlockManager.allocate / may_append 往 block_table 里填block编号
#                    │
#                    │ ModelRunner.prepare_prefill / prepare_decode：把 block_table 翻译成张量
#                    │   slot_mapping  [N]     每个新token写到哪个槽位（slot = block编号 × block_size + 块内偏移）
#                    │   block_tables  [B, M]  每条序列的block编号表，读KV时用
#                    │   用 pin_memory + .cuda(non_blocking=True) 从CPU拷贝到GPU（只拷几KB的整数）
#                    ▼
#   ┌───────────────────────────────── GPU 显存（每个 rank 各有一份） ─────────────────────────────────┐
#   │ ModelRunner.kv_cache  形状 [2, L, NB, BS, Hkv, D]                                                │
#   │   启动时在 allocate_kv_cache 中一次性分配（占满剩余显存），之后整个运行期间不再申请或释放        │
#   │   "分配/释放block" 只是 BlockManager 改了几个整数，显存里的数据原地不动，直接被覆盖              │
#   │                                                                                                  │
#   │   按层切出视图（不拷贝，共用同一块显存）：                                                       │
#   │     第0层  Attention.k_cache = kv_cache[0, 0]   形状 [NB, BS, Hkv, D]                            │
#   │            Attention.v_cache = kv_cache[1, 0]                                                    │
#   │     第1层  Attention.k_cache = kv_cache[0, 1]  ...                                               │
#   │     ...                                                                                          │
#   │                                                                                                  │
#   │   写：store_kvcache（Triton kernel）按 slot_mapping 把本次新token的K/V写进对应槽位               │
#   │   读：flash attention 按 block_tables 找到每条序列的各个block，读出全部历史K/V                   │
#   └──────────────────────────────────────────────────────────────────────────────────────────────────┘
#
# 张量并行（tp>1）时：
#   - 只有 rank 0 有 Scheduler/BlockManager；block_table 随 Sequence 通过共享内存发给其他rank
#   - 每个rank都有自己的 kv_cache，形状相同、block编号含义相同，但只存自己负责的那 Hkv = 总kv头数/tp 个头
#   - 例如 tp=2：block 7 在 rank 0 上存 kv头0~3，在 rank 1 上存 kv头4~7
#
# 二、kv_cache 的数据格式
#
#   形状 [2, L, NB, BS, Hkv, D]，每一维的含义：
#     维度     含义                         取值范围          Qwen3-0.6B（tp=1）
#     2        K 还是 V                     0=K，1=V          2
#     L        第几层                       0 ~ L-1           28
#     NB       物理block编号                0 ~ NB-1          由剩余显存决定，例如2450
#     BS       block内第几个token（偏移）   0 ~ BS-1          256
#     Hkv      第几个kv头                   0 ~ Hkv-1         8
#     D        头内第几个数                 0 ~ D-1           128
#   dtype 与模型一致（如 bfloat16，每个数2字节）
#
#   树状展开（每一层括号代表一个维度）：
#     kv_cache
#     ├─ [0] K
#     │   ├─ [第0层]
#     │   │   ├─ [block 0]
#     │   │   │   ├─ [偏移 0]   -> [Hkv, D]：这个槽位上的token，在第0层、所有kv头的K向量
#     │   │   │   ├─ [偏移 1]   -> [Hkv, D]
#     │   │   │   ├─ ...
#     │   │   │   └─ [偏移 255] -> [Hkv, D]
#     │   │   ├─ [block 1]  ...
#     │   │   └─ [block NB-1]
#     │   ├─ [第1层]  ...
#     │   └─ [第L-1层]
#     └─ [1] V（结构与K完全相同）
#
#   读一个元素的例子：kv_cache[0, 5, 3, 44, 2, :]
#     = 第5层、物理block 3、块内偏移44、kv头2 的 K 向量（128个数）
#     对于 block_table=[7, 3] 的序列，物理block 3 是它的第1个逻辑block，
#     所以这是该序列第 1×256 + 44 = 300 个位置（从0数）的token的K
#
#   单层 k_cache [NB, BS, Hkv, D] 在显存中的排布（连续存放，可以看作 [NB×BS 个槽位, Hkv×D]）：
#
#     槽位slot:  0      1     ...    255 │ 256    257   ...   511 │ 512  ...
#                └────── block 0 ───────┘  └────── block 1 ──────┘
#     每个槽位:  [kv头0：128个数][kv头1：128个数] ... [kv头7：128个数]  = 1024个bf16 = 2KB
#
#     slot 只与"物理block编号 + 块内偏移"有关，与是哪条序列无关；所以写入只需要一个 slot_mapping
#
#   显存占用估算（Qwen3-0.6B，bf16，tp=1）：
#     一个token、一层、K：     Hkv × D × 2字节      = 8 × 128 × 2      = 2KB
#     一个token、所有层、K和V： 2 × L × 2KB          = 2 × 28 × 2KB     = 112KB
#     一个block（256个token）：256 × 112KB                             = 28MB
#
#   kv_cache 里没有的东西：token id（只在CPU上的 Sequence/Block 里）、Q（用完即弃，不需要缓存）
#
# 三、一条请求的完整走查（block_size=256，prompt 300个token，无前缀缓存）
#
#   (1) 调度 prefill：BlockManager.allocate
#         需要 ceil(300/256) = 2 个block，从 free_block_ids 左边取出两个空闲block（假设是 7 和 3）
#         seq.block_table = [7, 3]
#
#   (2) prepare_prefill：计算每个token的写入槽位
#         token   0 ~ 255 -> block 7 -> slot 7×256+0 ~ 7×256+255 = 1792 ~ 2047
#         token 256 ~ 299 -> block 3 -> slot 3×256+0 ~ 3×256+43  =  768 ~  811
#         slot_mapping = [1792, 1793, ..., 2047, 768, 769, ..., 811]（共300个）
#
#   (3) 模型前向，28层中的每一层：
#         store_kvcache：本层算出的 K/V [300, Hkv, D] 按 slot_mapping 写入 k_cache/v_cache
#           第i个token -> k_cache 展平后的第 slot_mapping[i] 行
#         flash_attn_varlen_func：prefill 无前缀缓存时直接用本次的 k/v 计算注意力
#
#   (4) postprocess：采样得到第1个新token，追加到序列，len = 301
#
#   (5) 调度 decode：may_append
#         301 % 256 = 45 ≠ 1，最后一个block还有空位，不需要新block
#
#   (6) prepare_decode：
#         新token位置300，在第1个逻辑block（物理block 3）的偏移44
#         slot_mapping = [3×256 + 44] = [812]
#         context_lens = [301]，block_tables = [[7, 3]]
#
#   (7) 每一层：
#         store_kvcache：把这个新token的K/V写入 slot 812
#         flash_attn_with_kvcache：按 block_tables 读 block 7 的256个槽位 + block 3 的前45个槽位，
#           共 context_lens = 301 个位置的K/V，与新token的Q做注意力
#
#   (8) 序列结束：deallocate
#         block 3、7 的 ref_count 减到0，放回 free_block_ids；显存里的K/V不清除，
#         已填满的 block 7 的 hash 仍在 hash_to_block_id 里，之后相同前缀的请求可以直接复用
#====================================================================================================


# KV cache的最小管理单元（PagedAttention中的"页"）
# 一个block存放block_size个token在所有层的K和V，真正的显存在ModelRunner.kv_cache里，这里只记录元信息
class Block:

    def __init__(self, block_id):
        # 物理block编号，对应kv_cache张量第2维（从0数，即NB维）的下标，见文件开头的全局视角
        self.block_id = block_id
        # 引用计数：有多少条序列正在使用这个block
        # 前缀缓存命中时，多条序列可以共享同一个block，ref_count > 1
        self.ref_count = 0
        # 该block内容的hash（包含了前面所有block的前缀信息），-1表示未填满或未计算
        # 只有填满的block才会计算hash，未填满的block内容还会变化，不能被复用
        self.hash = -1
        # 该block存放的token，用于hash命中后再做一次精确比对，防止hash碰撞导致复用错误的KV
        self.token_ids = []

    # block被填满后，记录hash和token，使其可以被后续请求复用
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 分配给新序列时重置；ref_count直接置1，表示被分配者占用
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 负责KV block的分配、回收以及前缀缓存（prefix caching）
# 前缀缓存：多个请求有相同前缀时（如相同的system prompt），直接复用已经算好的KV，跳过这部分prefill计算
#
# 逻辑block与物理block（类比操作系统的虚拟内存分页，block_size=256）：
#
#   序列A  token   0~255  ─> 逻辑block0 ─> 物理block 7  <──┐
#          token 256~511  ─> 逻辑block1 ─> 物理block 3     │ 同一个物理block，ref_count=2
#          token 512~600  ─> 逻辑block2 ─> 物理block 12    │
#   序列B  token   0~255  ─> 逻辑block0 ─> 物理block 7  <──┘
#          token 256~300  ─> 逻辑block1 ─> 物理block 5
#
#   A.block_table = [7, 3, 12]      B.block_table = [7, 5]
#   物理block编号不需要连续，也不需要按顺序，空闲队列里取到哪个就用哪个
#   A和B前256个token完全相同（比如相同的system prompt），所以共享物理block 7，B不用再算一遍这部分KV
#
# 一个物理block的生命周期：
#
#     ┌────────────────────┐
#     │ (1) 空闲，无内容   │ <──────────────────────────────────────────────────┐
#     └────────────────────┘                                                    │
#               │ _allocate_block（新序列需要新block）                          │
#               ▼                                                               │
#     ┌────────────────────┐                                                    │
#     │ (2) 使用中，未填满 │ ref_count减到0 ──> 回到 (1)                        │
#     │    hash = -1       │                                                    │
#     └────────────────────┘                                                    │
#               │ hash_blocks（block被写满，计算hash并登记）                    │
#               ▼                                                               │
#     ┌────────────────────┐  ref_count减到0   ┌─────────────────────────┐      │
#     │ (3) 使用中，已填满 │ ────────────────> │ (4) 空闲，但内容仍有效  │ ─────┘
#     │   可被其他序列共享 │ <───────────────  │   hash仍在登记表中      │ 被当作新block分配出去，
#     └────────────────────┘ allocate命中缓存  └─────────────────────────┘ 删除hash登记，回到 (1)
#                            （从free队列捞回）
class BlockManager:

    # num_blocks：物理block总数（ModelRunner根据剩余显存算出的num_kvcache_blocks）
    # block_size：每个block容纳的token数
    def __init__(self, num_blocks: int, block_size: int):
        # 每个block容纳的token数
        self.block_size = block_size
        # 所有物理block的元信息，下标即block_id
        # self.blocks: list[Block] 是带类型注解的赋值，注解只是给人和IDE看的提示
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash -> 物理block id，用于查找可复用的block（前缀缓存的索引）
        # 注意：block被释放（ref_count=0）后，这里的映射不会立即删除，
        # 只要block还没被重新分配出去，里面的KV就仍然有效，后续请求依然可以命中
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲block队列，从左侧取、从右侧还，近似LRU：
        # 最近释放的block排在最后才会被覆盖，保留其缓存内容的时间最长
        # deque（双端队列）：两端插入/删除都是O(1)；普通list的 pop(0) 需要移动所有元素，是O(n)
        #   popleft() 从左边取出，append() 从右边放入
        # deque(range(3)) -> deque([0, 1, 2])
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 正在被使用（ref_count > 0）的block集合，用set是为了O(1)判断某个block是否在用
        # 不变量：free_block_ids 与 used_block_ids 互不相交，并集为全部block
        self.used_block_ids: set[int] = set()

    # 计算一个block的hash，把前一个block的hash作为前缀一起计算（链式hash）
    # 这样hash相同即意味着"从序列开头到当前block为止的所有token都相同"，
    # 而不仅仅是当前block的256个token相同；KV依赖于前面所有token，只有整个前缀相同才能复用
    # 例如：
    #   序列A：[sys prompt][问题1]   序列B：[sys prompt][问题2]
    #   第0个block的hash相同 -> 可复用；第1个block的hash不同 -> 需要重新计算
    #
    #   block0的hash: h0 = hash(          block0的token)
    #   block1的hash: h1 = hash(h0     +  block1的token)
    #   block2的hash: h2 = hash(h1     +  block2的token)
    #   h2 隐含了 block0、block1、block2 全部的token信息
    #
    # xxhash：非加密hash，速度非常快
    # @classmethod：类方法，第一个参数是类本身（cls）而不是实例（self），可以通过 BlockManager.compute_hash(...) 直接调用
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 创建一个64位hash计算器，可以多次update喂入数据，最后取结果
        h = xxhash.xxh64()
        if prefix != -1:
            # int.to_bytes(8, "little")：把整数转成8字节的二进制（小端序），hash只能处理字节
            h.update(prefix.to_bytes(8, "little"))
        # list[int] -> numpy数组 -> 原始字节（每个int64占8字节，256个token即2048字节）
        h.update(np.array(token_ids).tobytes())
        # 以整数形式返回hash值
        return h.intdigest()

    # 从空闲队列取一个block分配出去（不复用其中的内容）
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这个block即将被覆盖，原有内容失效，需要把它从hash表中删掉
        # 额外判断hash_to_block_id[hash] == block_id：同一个hash可能已被更新为指向其他block，那种情况下不能误删
        # dict.get(key)：key不存在时返回None而不是报错（dict[key] 在key不存在时会抛KeyError）
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    # 回收block：放回空闲队列尾部
    # 不清除hash和token_ids，被重新分配之前仍可以被前缀缓存命中
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 判断能否为一条新序列（prefill）分配所需的block
    # 返回值：-1 表示空闲block不足；>=0 表示可以分配，且前缀缓存命中了多少个block
    # 例如：序列需要4个block，前2个命中缓存，其中1个正被其他序列使用、1个在空闲队列中
    #   num_cached_blocks=2，num_new_blocks=4-1=3（在用的那个直接共享，不消耗空闲block）
    #   空闲block >= 3 时返回2，否则返回-1
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 只检查前num_blocks-1个block，最后一个block一定不复用，原因：
        #   1. 最后一个block可能没填满，没有hash
        #   2. 即使prompt全部命中缓存，也至少要留一个token参与计算，才能得到最后位置的logits来采样下一个token
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            # dict.get(key, default)：key不存在时返回default，这里不存在返回-1
            block_id = self.hash_to_block_id.get(h, -1)
            # 未命中，或者hash碰撞（内容不一致），后面的block也不可能命中了（链式hash），直接退出
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 命中的block如果正被其他序列使用，直接共享，不需要占用新的空闲block
            # 如果命中的block在空闲队列中（已被释放但内容还在），则需要把它从空闲队列"拿回来"，依然消耗一个空闲block名额
            # x in set：判断元素是否在集合中，O(1)；同样的操作对list是O(n)
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    # 为序列分配block，前num_cached_blocks个复用缓存，其余分配新的
    # 必须在can_allocate返回>=0之后调用
    #
    # 完整示例（block_size=4，方便演示）：
    #   之前序列X算过 [1,2,3,4 | 5,6,7,8 | 9]，已登记 hash(1234)->block 7，hash(5678)->block 3
    #   新序列Y = [1,2,3,4 | 5,6,7,8 | 0,0,0]，num_blocks=3
    #   can_allocate：检查前2个block，都命中 -> num_cached_blocks=2
    #   allocate：
    #     block_table = [7, 3]          （复用，ref_count+1）
    #     block_table = [7, 3, 新block]  （第3个block新分配）
    #     num_cached_tokens = 2 * 4 = 8  -> prefill只需计算最后3个token
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        # 复用前缀缓存命中的block
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 正在被其他序列使用，增加引用计数即可
                block.ref_count += 1
            else:
                # 在空闲队列中，把它取出来重新激活，内容保持不变
                # deque.remove是O(n)，但只在命中缓存时发生，可以接受
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 未命中的部分分配全新的block
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 命中部分的KV已经在cache中，prefill时可以跳过这些token
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 释放序列占用的所有block（序列结束或被抢占时调用）
    # 逆序释放：靠后的block先进入空闲队列，先被覆盖；
    # 靠前的block（公共前缀，复用价值更高）后进入空闲队列，在缓存中保留得更久
    def deallocate(self, seq: Sequence):
        # reversed(list)：从后往前遍历，不修改原列表
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 引用计数归零才真正回收，共享的block要等所有使用者都释放
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # decode前判断能否继续追加一个token
    # 本次decode的输入是last_token，位置为len(seq)-1
    # 当len(seq) % block_size == 1时，说明last_token是新block的第一个token，需要一个新的空闲block
    # 例如：block_size=256，len(seq)=257，第257个token（位置256）要写到第2个block
    #
    #   len(seq)   last_token位置   位置所在block   是否需要新block（block_size=256）
    #     256          255             第0个          否（第0个block已存在）
    #     257          256             第1个          是（len % 256 == 1）
    #     258          257             第1个          否（上一步已分配）
    #
    # 这里利用了 True == 1、False == 0：需要新block时要求空闲数>=1，否则>=0（恒成立）
    # %：取余，257 % 256 = 1
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 如果需要，为序列追加一个新block（与can_append的条件对应）
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 在一个step计算完成之后调用：为本次刚好填满的block计算hash并登记到hash表，供后续请求复用
    # 只登记已经填满、且KV已写入cache的block
    def hash_blocks(self, seq: Sequence):
        # [start, end)：本次step之后新填满的block范围
        # 例如：block_size=256，num_cached_tokens=200，num_scheduled_tokens=400
        #   start = 200 // 256 = 0，end = 600 // 256 = 2 -> 第0、1个block新填满
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 链式hash需要前一个block的hash作为前缀
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
