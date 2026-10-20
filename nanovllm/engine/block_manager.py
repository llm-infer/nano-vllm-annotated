from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


# KV cache的最小管理单元（PagedAttention中的"页"）
# 一个block存放block_size个token在所有层的K和V，真正的显存在ModelRunner.kv_cache里，这里只记录元信息
class Block:

    def __init__(self, block_id):
        # 物理block编号，对应kv_cache张量第3维的下标
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
class BlockManager:

    # num_blocks：物理block总数（ModelRunner根据剩余显存算出的num_kvcache_blocks）
    # block_size：每个block容纳的token数
    def __init__(self, num_blocks: int, block_size: int):
        # 每个block容纳的token数
        self.block_size = block_size
        # 所有物理block的元信息，下标即block_id
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash -> 物理block id，用于查找可复用的block（前缀缓存的索引）
        # 注意：block被释放（ref_count=0）后，这里的映射不会立即删除，
        # 只要block还没被重新分配出去，里面的KV就仍然有效，后续请求依然可以命中
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲block队列，从左侧取、从右侧还，近似LRU：
        # 最近释放的block排在最后才会被覆盖，保留其缓存内容的时间最长
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
    # xxhash：非加密hash，速度非常快
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 从空闲队列取一个block分配出去（不复用其中的内容）
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这个block即将被覆盖，原有内容失效，需要把它从hash表中删掉
        # 额外判断hash_to_block_id[hash] == block_id：同一个hash可能已被更新为指向其他block，那种情况下不能误删
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
            block_id = self.hash_to_block_id.get(h, -1)
            # 未命中，或者hash碰撞（内容不一致），后面的block也不可能命中了（链式hash），直接退出
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 命中的block如果正被其他序列使用，直接共享，不需要占用新的空闲block
            # 如果命中的block在空闲队列中（已被释放但内容还在），则需要把它从空闲队列"拿回来"，依然消耗一个空闲block名额
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    # 为序列分配block，前num_cached_blocks个复用缓存，其余分配新的
    # 必须在can_allocate返回>=0之后调用
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
    # 这里利用了 True == 1、False == 0：需要新block时要求空闲数>=1，否则>=0（恒成立）
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
