from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


# 序列的生命周期状态
# WAITING：在等待队列中，还没做完prefill（新请求、chunked prefill做了一半、被抢占的请求）
# RUNNING：prefill已完成，在运行队列中逐token decode
# FINISHED：遇到EOS或达到max_tokens，KV block已释放
class SequenceStatus(Enum):
    # auto()：自动给枚举成员分配值（1、2、3...），只关心名字不关心值时使用
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# 一条请求在引擎内部的表示：保存token、状态、KV block映射以及采样参数
class Sequence:
    # 类变量，所有实例共享；LLMEngine初始化时会用config.kvcache_block_size覆盖
    block_size = 256
    # 全局自增计数器，为每条序列分配唯一的seq_id
    # generate结束时按seq_id排序，使输出顺序与输入prompt顺序一致
    counter = count()

    # 注意：默认参数SamplingParams()只在函数定义时创建一次，所有未传参的调用共享同一个对象
    # 这里只读取它的字段、不会修改，所以没有问题
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # 浅拷贝，后续append_token不会影响调用方传进来的list
        self.token_ids = copy(token_ids)
        # 最后一个token，decode时作为模型输入
        self.last_token = token_ids[-1]
        # 当前总token数（prompt + 已生成）
        self.num_tokens = len(self.token_ids)
        # prompt的token数，用于区分prompt和生成部分
        self.num_prompt_tokens = len(token_ids)
        # 已经算好KV并写入KV cache的token数
        # 来源有两种：前缀缓存命中的部分 + 之前step已经计算过的部分（chunked prefill / decode）
        self.num_cached_tokens = 0
        # 本次step被调度计算的token数
        # prefill：本次处理的prompt片段长度（chunked prefill时可能小于剩余长度）
        # decode：固定为1
        self.num_scheduled_tokens = 0
        # 是否处于prefill阶段，影响跨进程序列化时传输的内容（见__getstate__）
        self.is_prefill = True
        # 逻辑block -> 物理block的映射表，block_table[i]是第i个逻辑block对应的物理block id
        # 类似操作系统的页表，物理block在显存中不必连续
        self.block_table = []
        # 采样参数直接拷贝到序列上，便于ModelRunner批量读取
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # 支持len(seq)，返回总token数
    def __len__(self):
        return self.num_tokens

    # 支持seq[i]、seq[a:b]，直接索引token_ids
    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 已生成的token数
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 当前token数需要占用多少个block，向上取整
    # 例如：block_size=256，num_tokens=600 -> 3个block
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一个block里实际存了多少个token，取值范围[1, block_size]
    # 例如：block_size=256，num_tokens=600 -> 600 - 2*256 = 88
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 取第i个逻辑block对应的token_ids，用于计算前缀缓存的hash
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # decode每生成一个token就追加到序列末尾
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 自定义pickle序列化
    # 张量并行时，rank0需要通过共享内存把序列发送给其他rank，序列化的数据越小越好
    # 其他rank只负责模型前向，不做调度和采样，所以只发送前向需要的字段（不含seq_id、temperature等）
    # prefill：需要完整的token_ids来构造输入
    # decode：只需要最后一个token，避免每个step都传输整条序列，长序列时可省下大量序列化开销
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # 反序列化，与__getstate__对应
    # 通过last_state的类型判断发送方处于prefill（list）还是decode（int）
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode阶段只拿到最后一个token，token_ids置空；
            # prepare_decode只用到last_token、num_tokens、block_table，不受影响
            self.token_ids = []
            self.last_token = last_state
