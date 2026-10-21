from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


# 序列的生命周期状态
# auto()：自动给枚举成员分配值（1、2、3...），只关心名字不关心值时使用
# 状态流转图：
#
#   add_request
#       │
#       ▼
#   ┌─────────┐   prefill算完整个prompt     ┌─────────┐   生成EOS或达到max_tokens     ┌──────────┐
#   │ WAITING │ ─────────────────────────> │ RUNNING │ ──────────────────────────> │ FINISHED │
#   └─────────┘                            └─────────┘                             └──────────┘
#     ▲   │ chunked prefill没算完，                │
#     │   └─ 留在WAITING，下一步继续                │
#     └──────────────────────────────────────────┘
#           KV cache不够，被抢占(preempt)，释放所有block，回到WAITING队头，之后重新prefill
class SequenceStatus(Enum):
    # 在等待队列中，还没做完prefill（请求刚进来、chunked prefill做了一半、资源被抢占）
    WAITING = auto()
    # prefill已完成，在运行队列中逐token decode
    RUNNING = auto()
    # 遇到EOS或达到max_tokens，KV block已释放
    FINISHED = auto()


# 一条请求在引擎内部的表示：保存token、状态、KV block映射和采样参数
# 各计数字段变化示例（block_size=256, prompt长600，前缀缓存命中1个block，num_max_batched_tokens=200）
#   初始化：            num_tokens=600，num_prompt_tokens=600，num_cached_tokens=0，block_table=[]
#   第1次调度：          num_cached_tokens=256（命中1个block），block_table=[5,9,12]（三个block分别存放256+256+88=600个token）
#                      num_scheduled_tokens=200（预算只有200，chunked prefill）
#   第1次postprocess：  num_cached_tokens=456，num_scheduled_tokens=0，还没prefill完成，丢弃采样结果
#   第2次调度：          num_scheduled_tokens=144（剩余600-456），状态变为RUNNING
#   第2次postprocess：  num_cached_tokens=600，追加第1个生成token -> num_tokens=601
#   之后每次decode：     num_scheduled_tokens=1，num_cached_tokens+1，num_tokens+1
class Sequence:
    # 类变量，所有实例共享
    # 读取self.block_size时，实例上没有，就去类上找
    block_size = 256
    # 全局自增器，为每条序列分配唯一的id
    counter = count()

    # 默认SamplingParams()只在函数定义时创建一次，所有未传参的调用共享一个SamplingParams()
    # 因为是只读sampling_params的字段，所以共享没有问题
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 从0开始自增，也代表提交顺序
        self.seq_id = next(Sequence.counter)
        # 新序列状态为waiting
        self.status = SequenceStatus.WAITING
        # 全部的token id：prompt+已生成的token
        # 浅拷贝：后续append_token不影响调用方传进来的list
        self.token_ids = copy(token_ids)
        # 最后一个token，decode时作为输入，计算KV
        self.last_token = token_ids[-1]
        # 当前总token数（prompt + 已生成）
        self.num_tokens = len(self.token_ids)
        # prompt的token数，区分prompt和生成部分
        self.num_prompt_tokens = len(token_ids)
        # 已算好kv并写入kv cache的token数
        # 来源：前缀缓存命中的部分+本序列step计算过的部分（prefill/decode）
        self.num_cached_tokens = 0
        # 本次step需要计算的token数
        # prefill：本次处理的prompt长度（chunked prefill时可能小于prompt剩余长度）
        # decode：固定为1
        self.num_scheduled_tokens = 0
        # 是否处在prefill阶段
        self.is_prefill = True
        # kv 逻辑block -> kv 物理block映射表
        # block_table[i]代表第i个逻辑block对应的物理block id
        # 类似操作系统的页表，物理block在显存中不必连续
        # 例如：block_table=[5,9,12]，表示第0~255个kv在物理block 5,第256~511个kv在物理block 9
        self.block_table = []
        # 请求可配置采样参数，将其拷贝到序列上，便于ModelRunner批量读取
        self.temperature = sampling_params.temperature
        # 此序列允许生成token的上限，达到上限后直接结束
        self.max_tokens = sampling_params.max_tokens
        # 是否忽略EOS继续生成
        self.ignore_eos = sampling_params.ignore_eos

    # 以双下划钱开头和结尾的方法叫‘魔法方法’，会在特定语法下自动调用
    # len(seq) 等价于 seq.__len__()，返回token总数
    def __len__(self):
        return self.num_tokens

    # 获取某个或某段token，例如：seq[3] -> self.token_ids[3]；seq[0:5] -> self.token_ids[0:5]
    # key可以是下标或切片对象
    def __getitem__(self, key):
        return self.token_ids[key]

    # @property把方法变成‘只读属性’
    # 适用于由其他字段计算得到的值，每次访问都重新计算
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 已生成的token数，判断是否达到max_tokens
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # 已生成的token，用于detokenizer生成文本，返回客户端
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 当前token数需要占用多少kv block，向上取整
    # 新请求需占用block = num_blocks - 命中缓存的block数
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一个block存了多少token = 总token数 - 前num_blocks-1 个满block存的token数
    # 用于计算当前token存储在物理block上的偏移量
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 取第i个逻辑block对应的token ids，用于计算前缀缓存的hash
    # 超出长度时自动截断，例如block_size=256，num_tokens=600
    # block(2) -> token_ids[512,768]，实际只有88个元素，即token_ids[512,599]
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # decode每生成一个token，就追加到序列末尾
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 自定义序列化字段
    # python序列化库pickle默认序列化所有字段（self.__dict__）
    # 定义了__getstate__后，pickle只序列化它的返回值；还原时把这个值传给__setstate__
    # 传输路径：rank 0负责调度指挥，把seq派发给其他rank
    #    rank 0：pickle.dumps(seq) -> 调用 seq.__getstate__() -> 字符串 -> 共享内存
    #    rank 1：共享内存 -> 字符串 -> pickle.loads -> 调用 __setstate__ -> seq
    def __getstate__(self):
        # prefill：需要完整的token_ids来计算下一个token
        # decode：历史token的kv存在kv cache里，只需要最后一个token，避免每个step都传输整条序列，节省序列化开销
        last_state = self.last_token if not self.is_prefill else self.token_ids
        # rank只做前向计算，所以只发送前向需要的数据即可
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # 反序列化
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        # prefill阶段last_state是完整的token_ids
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode阶段last_state是上一个step生成的token id
            self.token_ids = []
            self.last_token = last_state
