from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# 调度器，决定每step计算哪些seq，各计算多少token
# 策略：
#   1. prefill优先：只要等待队列里有能调度的请求，本step就只做prefill
#   2. 没有prefill可做时，对所有running的序列做一次decode，各生成1个token
#   3. 同一个step里不会混合prefill和decode
#   4. KV cache不够时，抢占最晚加入running的序列，释放其block，放回waiting队列等待重新prefill
# 流程图：
#          add()                    prefill算完                     结束(EOS/max_tokens)
#   新请求 ───────>  ┌─────────┐  ────────────────────> ┌─────────┐  ──────────────────> 释放block，移出
#                   │ waiting │                        │ running │
#                   └─────────┘  <──────────────────── └─────────┘
#                     ▲      │   preempt：放回waiting队头（appendleft）
#                     └──────┘
#                   chunked prefill未完成：留在waiting队头，下一步继续
# 决策流程：
#   ┌────────────────────── (1) prefill 阶段 ──────────────────────┐
#   │ while waiting非空 且 未达max_num_seqs:                        │
#   │     seq = waiting队头                                        │
#   │     剩余token预算为0？是->跳出循环                              │
#   │     首次调度 且 KV block不够？是->跳出循环                       │
#   │     预算放不下整个seq 且 不是第一条？是->跳出循环                  │
#   │     分配block（仅首次）                                        │
#   │     本次计算min(剩余要算的token数，剩余预算)个token               │
#   │     prompt全部算完？是->移入running队列                         │
#   │     加入scheduled_seqs                                       │
#   └──────────────────────────────────────────────────────────────┘
#                      │
#          scheduled_seqs非空？ ── 是 ──> return (scheduled_seqs, True)
#                      │ 否（没有能做的prefill）
#                      ▼
#   ┌────────────────────── (2) decode 阶段 ───────────────────────┐
#   │ while running非空 且 未达max_num_seqs:                        │
#   │     seq = 从running队头取出                                   │
#   │     while KV block不够追加1个token:                           │
#   │         running还有序列？ 是 -> 抢占running队尾的序列            │
#   │                          否 -> 抢占seq自己，跳过它             │
#   │     （block够了）加入scheduled_seqs                           │
#   └─────────────────────────────────────────────────────────────┘
#                      │
#                      ▼
#   scheduled_seqs放回running队头，return (scheduled_seqs, False)
class Scheduler:

    def __init__(self, config: Config):
        # 单step最多调度的序列数，主要影响decode性能
        #   相较于prefill，decode持续时间长，并发度很容易超过max_num_seqs
        #   decode step阶段读取一次权重，增加max_num_seqs，可以把权重读取成本摊薄
        #   max_num_seqs存在上限，由可用的kv block决定
        self.max_num_seqs = config.max_num_seqs
        # 单step最多计算的token数（主要约束prefill；decode每条序列1个token）
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 结束符的token id，由tokenizer提供
        self.eos = config.eos
        # 每个kv block可容纳的token数
        self.block_size = config.kvcache_block_size
        # num_kvcache_blocks在ModelRunner初始化时，根据剩余显存自动计算得到
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待prefill 的序列（新请求、chunked prefill未完成的、被抢占的）
        self.waiting: deque[Sequence] = deque()
        # 已完成prefill，处于decode阶段的序列
        self.running: deque[Sequence] = deque()

    # 空容器在条件判断中被当做False
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 返回本step需要计算的seq，以及是否为prefill
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        # 记录本step消耗的token预算
        num_batched_tokens = 0

        # prefill
        # 从waiting中取序列
        # prefill优先调度：每次调度优先看waiting队列
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 只看队头，不取出；能算完时才真正取出
            seq = self.waiting[0]
            # 本step剩余的token预算
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            # 第一次被调度，或者被抢占后重新调度，此时还没申请block_table
            if not seq.block_table:
                # 检查block是否足够，并查询缓存命中情况
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # 返回-1，代表空闲block不够；队列后面的seq也不调度了，保持FIFO，防止队头饿死
                if num_cached_blocks == -1:
                    break
                # 本次step需要计算的token = 总数 - 前缀缓存命中的token数
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已经分配过block，说明是chunked prefill的后续块，继续计算剩余部分
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 剩余token预算不够用来调度此seq，而且前面已经有序列被选中，则把此seq留到下一个step
            # 为啥不用chunked prefill调度此seq？scheduled_seqs不为空，
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
             # 一次性为整条序列分配所有block（包括本次还没算到的chunk）
             # 传入num_cached_blocks，不再为缓存的token申请block
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本次实际计算的token数，预算不足时只算一部分
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 本次能把prompt全部算完，则移入running队列，下一个step开始decode
            # 否则继续留在waiting队头，下一个step继续算下一个chunk
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # 如果有可调度的prefill请求，则先执行prefill。prefill优先体现在这里
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 每个running的序列各计算1个token
        # 抢占示例（running=[A,B,C,D]，空闲block为0，每条序列刚好都需要新block）
        #   取A：block不够 -> 抢占队尾D（释放D的block）-> block够了 -> A被调度
        #   取B：block不够 -> 抢占队尾C -> 够了 -> B被调度
        #   running为空，退出循环，本轮调度A、B；C和D回到waiting队头等待重新prefill
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从队头中取出
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 空闲block不够，不断抢占running队尾的seq，直到block够用
                    self.preempt(self.running.pop())
                else:
                    # 没有别的seq可以抢了，只能抢占自己
                    self.preempt(seq)
                    break
            # block足够，本序列可以参与本次decode
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 检查是否需要申请新的block，追加到block_table
                # 当前block存满时，才需要追加
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 放回running队列，默认它们还会继续decode
        # 当seq遇到EOS或达到max_tokens时，会调用self.running.remove(seq)把它移除
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 抢占running状态的seq，释放kv block，放回waiting队列队头，下次schedule优先考虑它
    # 这里是重计算式抢占，而不是把KV换到CPU
    # 由于block释放后，hash仍保留，如果这些block没有被覆盖，重新prefill时仍有可能命中缓存
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 模型forward和sampling之后的后处理：更新seq状态，追加新token，判断是否结束
    # seqs：本step调度的序列；token_ids：本step生成的token，与seqs一一对应
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # zip把两个列表按位置配对：(seqs[0], token_ids[0]), (seq[1], token_ids[1]), ...
        for seq, token_id in zip(seqs, token_ids):
            # 把本次计算填满的block登记到前缀缓存
            self.block_manager.hash_blocks(seq)
            # 本次计算的token的KV已写入cache
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # chunked prefill还没算完整个prompt，此时计算出的token没有意义，直接丢弃
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # 结束条件：生成了EOS，或者达到最大长度
            # 释放kv block、从running队列中移除
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
