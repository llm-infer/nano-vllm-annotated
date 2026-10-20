from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# 调度器：每个step决定"本次让哪些序列参与计算、各计算多少个token"
# 策略：
#   1. prefill优先：只要等待队列里有能调度的请求，本step就只做prefill（降低TTFT）
#   2. 没有可做的prefill时，做一次decode，所有running序列各生成一个token
#   3. 同一个step里不会混合prefill和decode
#   4. KV cache不够时，抢占（preempt）最晚加入的running序列，释放其block，之后重新prefill
class Scheduler:

    def __init__(self, config: Config):
        # 单个step最多调度的序列数（prefill和decode都受限）
        self.max_num_seqs = config.max_num_seqs
        # 单个step最多计算的token数（主要约束prefill；decode每条序列1个token）
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 结束符token id，由tokenizer提供
        self.eos = config.eos
        # 每个KV block容纳的token数
        self.block_size = config.kvcache_block_size
        # num_kvcache_blocks在ModelRunner初始化时根据剩余显存计算得出
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待prefill的序列（新请求、chunked prefill未完成的、被抢占的）
        self.waiting: deque[Sequence] = deque()
        # 已完成prefill，处于decode阶段的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 返回：(本次调度的序列列表, 是否为prefill)
    # 每条被调度序列的num_scheduled_tokens会被设置为本step要计算的token数
    # 例如（max_num_batched_tokens=1000，waiting中有A(800个token)、B(500个token)，均无缓存）：
    #   A：remaining=1000 >= 800，调度800个，A全部算完，移入running
    #   B：remaining=200 < 500，且scheduled_seqs非空，不允许切块，break
    #   本step只做A；下个step B成为队头，可以被切块
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # 按FIFO顺序从等待队列取序列，直到达到序列数上限、token预算耗尽或KV block不足
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # 本step剩余的token预算
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 第一次被调度（或被抢占后重新调度）：检查block是否足够，并查询前缀缓存命中情况
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 空闲block不够，后面的序列也不调度了（保持FIFO，避免队头请求饿死）
                    break
                # 需要计算的token数 = 总数 - 前缀缓存命中的部分
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已经分配过block，说明是chunked prefill的后续块，继续计算剩余部分
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # chunked prefill：预算不够放下整条序列时，把它切成块分多个step处理
            # 只允许batch中的第一条序列被切块：如果前面已经有序列了，就留到下一个step，
            # 这样同一时刻最多只有一条序列处于"prefill了一半"的状态
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                # 一次性为整条序列分配所有block（包括本次还没算到的chunk）
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本次实际计算的token数，预算不足时只算一部分
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 本次能把prompt全部算完，就移入running队列，下个step开始decode
            # 否则留在waiting队头，下个step继续算下一个chunk
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 每条running序列各计算1个token
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 空闲block不够追加新token时，抢占running队尾（最晚加入）的序列，释放它的block
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 没有别的序列可抢占了，只能抢占自己
                    self.preempt(seq)
                    break
            # while...else：while条件为False正常退出（没有被break）时才执行else
            # 即：block足够（可能是抢占其他序列之后），本序列可以参与本次decode
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把本次调度的序列按原顺序放回running队头，保持先来先服务的顺序
        # extendleft会逆序插入，所以先reversed一次
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 抢占：释放序列的全部KV block，放回等待队列队头，之后优先重新prefill
    # 这是"重计算"（recompute）式抢占，不把KV换出到CPU（swap）
    # 由于block释放后hash仍保留，如果这些block还没被覆盖，重新prefill时可以命中前缀缓存，减少重算量
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 模型前向+采样之后的后处理：更新序列状态、追加新token、判断是否结束
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # 本次计算填满的block登记到前缀缓存（需要在更新num_cached_tokens之前调用）
            self.block_manager.hash_blocks(seq)
            # 本次计算的token的KV已写入cache
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # chunked prefill还没算完整个prompt，此时采样出的token没有意义（不是最后位置的logits），直接丢弃
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # 结束条件：生成了EOS（且未设置ignore_eos），或者达到最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
