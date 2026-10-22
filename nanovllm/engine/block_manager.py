from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


# BlockManager：在CPU上记账（哪块block空闲、哪个被使用，内容hash），只记录元数据，不碰显存
# ModelRunner.kv_cache：GPU上预先分配的显存，真正存放KV数据的地方
#
# KV cache结构
#   kv_cache = torch.empty(2, L, NB, BS, Hkv, D)   GPU上的一个张量，启动时分配
#                          │  │  │    |   └───┴─ KV head 数，单个 token 单个头的向量
#                          │  │  │    └─ block内第几个token
#                          │  │  └─ 第几个block（block_table存的就是这个编号），block 是"跨层"的逻辑单位，BlockManager 分配一个 block，等于在所有层各占一页
#                          │  └─ 第几层（token在第i层持有的KV），2和L维度放在最外面，保证切出第i层的K是一块连续内存
#                          └─ 0=K，1=V
# 某一层的k_cache = kv_cache[0, 层号]，形状[NB, BS, Hkv, D], 例如NB=6，BS=4
#              偏移0    偏移1   偏移2    偏移3
#   block 0  [      ][      ][      ][      ]
#   block 1  [ A4   ][ A5   ][      ][      ] <- 序列A的第2个block
#   block 2  [      ][      ][      ][      ]
#   block 3  [ B0   ][ B1   ][ B2   ][      ] <- 序列B的第1个block
#   block 4  [ A0   ][ A1   ][ A2   ][ A3   ] <- 序列A的第1个block
#   block 5  [      ][      ][      ][      ]
# 序列A（6个token）：block_table=[4,1]，序列B（3个token）：block_table=[3]
#
# BlockManager
#   blocks[i]         Block(block_id=i, ref_count, hash, token_ids)  每个物理block的元信息
#   free_block_ids    deque([2, 5, 8, ...])   空闲的block编号
#   used_block_ids    {3, 7, ...}             正在使用的block编号   
#   hash_to_block_id  {hash: block_id}        前缀缓存索引   
# 
# 一条请求的完整流程（block_size=256，prompt 300个token，无前缀缓存）
#   BlockManager.allocate
#       需要 ceil(300/256) = 2 个block，从 free_block_ids 左边取出两个空闲block（假设是 7 和 3）
#       seq.block_table = [7, 3]
#   prepare_prefill：计算每个token的写入槽位
#       token   0 ~ 255 -> block 7 -> slot 7×256+0 ~ 7×256+255 = 1792 ~ 2047
#       token 256 ~ 299 -> block 3 -> slot 3×256+0 ~ 3×256+43  =  768 ~  811
#       slot_mapping = [1792, 1793, ..., 2047, 768, 769, ..., 811]（共300个）
#   模型前向，28层中的每一层：
#       第i个token -> k_cache展平后的第slot_mapping[i]行
#   postprocess：采样得到第1个新token，追加到序列，len = 301
#   调度 decode：may_append
#       301 % 256 = 45 ≠ 1，最后一个block还有空位，不需要新block
#   prepare_decode
#       新token位置300，在第1个逻辑block（物理block 3）的偏移44
#       slot_mapping = [3×256 + 44] = [812]
#       context_lens = [301]，block_tables = [[7, 3]]
#   每一层
#       把新token的kv写入slot 812
#   结束
#       block 3、7 的 ref_count减到0，放回free_block_ids，显存里的KV不清除
#       已填满的block 7的hash仍在hash_to_block_id里，相同前缀请求可以直接复用
# KV cache的最小管理单元
# 一个block存放block_size个token在所有层的K和V
class Block:

    def __init__(self, block_id):
        # 物理block编号，对应kv_cache张量第2维的下标
        self.block_id = block_id
        # 被多少seq使用
        # 前缀缓存命中时，多条seq可以共享一个block
        self.ref_count = 0
        # block内容的hash，包含了前面所有block的前缀信息
        # 只有填满的block才会计算hash，未填满的block内容还会变化，不能被检索、复用
        self.hash = -1
        # 该block存放的token，用于hash命中后再做一次精确比对，防止hash碰撞导致复用错误的KV
        self.token_ids = []

    # block被填满后，记录hash和token，便于后续请求复用
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 分配给新序列时，重置元数据
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 负责KV block的分配、回收、前缀查找
# 前缀缓存：多个请求有相同的前缀，直接复用已经算好的KV
# 逻辑block和物理block：
#   序列A  token   0~255  ─> 逻辑block0 ─> 物理block 7  <──┐
#          token 256~511  ─> 逻辑block1 ─> 物理block 3     │ 同一个物理block，ref_count=2
#          token 512~600  ─> 逻辑block2 ─> 物理block 12    │
#   序列B  token   0~255  ─> 逻辑block0 ─> 物理block 7  <──┘
#          token 256~300  ─> 逻辑block1 ─> 物理block 5
#   A.block_table = [7,3,12]，B.block_table=[7,5]
class BlockManager:

    # num_blocks：物理block总数（ModelRunner根据剩余显存算出的num_kvcache_blocks）
    # block_size：每个block容纳的token数
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash -> 物理block id，用于查找可复用的block
        # block被释放后，到被重新分配之前，映射不会删除，里面的KV依旧有效，后续请求依然可以命中
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲block队列，左侧取，右侧还
        # 最近释放的block最晚被重新分配，使其保留的缓存时间最长
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 正在使用的block
        self.used_block_ids: set[int] = set()

    # 计算一个block的hash，把前一个block的hash作为block的一部分一起计算（链式hash）
    # 如果两个block的hash相同，意味着它们前面的block也相同
    # 例如：
    #   序列A：[sys prompt][问题1]   序列B：[sys prompt][问题2]
    #   第0个block的hash相同 -> 可复用；第1个block的hash不同 -> 需要重新计算
    #
    #   block0的hash: h0 = hash(          block0的token)
    #   block1的hash: h1 = hash(h0     +  block1的token)
    #   block2的hash: h2 = hash(h1     +  block2的token)
    #   h2 隐含了 block0、block1、block2 全部的token信息
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 创建一个64位的hash计算器，可多次update喂入数据
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 从空闲队列里取一个block
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这个block即将被覆盖，需要把对应的KV cache抹掉，防止被读取到脏数据
        # KV cache只有当缓存命中时，才会被读取到，所以只需把hash_to_block_id记录的映射删掉即可
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    # 回收block：放回空闲队列尾部
    # 不清除hash和token_ids，被重新分配之前还可以被前缀缓存命中
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 判断能否为一条新序列（prefill）分配所需block
    # -1：空闲block不足
    # >=0：可以分配，返回前缀缓存命中了多少个block
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 只检查前num_blocks - 1个block，最后一个block一定不能复用
        #   最后一个block可能没填满，没有hash
        #   即使填满了，也不能复用，因为最后一个位置必须真正跑一遍模型。既然复用以整块为单位，最后一个 block 就整块不复用
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 未命中，或者hash碰撞（内容不一致），后面的block也不可能命中了，直接退出
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 如果命中的block正在被其他序列使用，直接共享，不需要申请新的block
            # 如果命中的block在空闲队列中，需要把它从空闲队列拿回來，依然消耗一个block名额
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 如果剩余block不够，返回-1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    # 为序列分配block，前num_cached_blocks个block直接复用，其余的重新分配
    # 前提是can_allocate()返回大于0
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        # 复用前num_cached_blocks个block
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 复用的block已经被其他序列使用，增加引用计数即可
                block.ref_count += 1
            else:
                # 复用的block在空闲队列中，重新激活，内容保持不变
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 未命中的部分需要重新分配
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 计算prefill可跳过的token数量
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 释放序列占用的block（结束或被抢占）
    # 逆序释放：序列中越靠前的block，复用价值越高，越晚进入队列，在队列中存活时间越长，被重新激活的概率越高
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 引用数归零后，才真正回收
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # decode开始前，判断有空间追加一个token
    def can_append(self, seq: Sequence) -> bool:
        # 如果不需要新block，则直接返回true
        # 需要新block时，要求空闲block>=1
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 如果需要，为序列追加一个新block
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 为本次刚好填满的block计算hash，并注册到hash表里，供后续请求复用
    # 每次step完成以后调用
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
