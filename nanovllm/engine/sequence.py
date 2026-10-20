from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


# 序列的生命周期状态
# WAITING：在等待队列中，还没做完prefill（新请求、chunked prefill做了一半、被抢占的请求）
# RUNNING：prefill已完成，在运行队列中逐token decode
# FINISHED：遇到EOS或达到max_tokens，KV block已释放
#
# 状态流转图：
#
#   add_request
#       │
#       ▼
#   ┌─────────┐  prefill算完整个prompt     ┌─────────┐  生成EOS或达到max_tokens    ┌──────────┐
#   │ WAITING │ ─────────────────────────> │ RUNNING │ ──────────────────────────> │ FINISHED │
#   └─────────┘                            └─────────┘                             └──────────┘
#     ▲   │                                     │
#     │   └─ chunked prefill没算完：            │
#     │        留在WAITING，下一步继续          │
#     └─────────────────────────────────────────┘
#        被抢占(preempt)：KV cache不够，释放所有block，回到WAITING队头，之后重新prefill
#
# Enum（枚举）：定义一组有名字的常量，比直接用数字/字符串更不容易写错
#   SequenceStatus.WAITING == SequenceStatus.WAITING  -> True
#   SequenceStatus.WAITING == 1                       -> False（枚举成员不等于普通数字）
class SequenceStatus(Enum):
    # auto()：自动给枚举成员分配值（1、2、3...），只关心名字不关心值时使用
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# 一条请求在引擎内部的表示：保存token、状态、KV block映射以及采样参数
# 各计数字段的变化示例（block_size=256，prompt长600，前缀缓存命中1个block，max_num_batched_tokens=200）：
#   初始化：              num_tokens=600  num_prompt_tokens=600  num_cached_tokens=0    block_table=[]
#   第1次调度（allocate）：num_cached_tokens=256（命中1个block） block_table=[5, 9, 12]
#                         num_scheduled_tokens=200（预算只有200，chunked prefill）
#   第1次postprocess：    num_cached_tokens=456  num_scheduled_tokens=0，采样结果丢弃
#   第2次调度：           num_scheduled_tokens=144（剩余600-456），状态变为RUNNING
#   第2次postprocess：    num_cached_tokens=600，追加第1个生成token -> num_tokens=601
#   之后每次decode：      num_scheduled_tokens=1，postprocess后num_cached_tokens+1、num_tokens+1
class Sequence:
    # 类变量：写在类体中、方法之外的变量，属于类本身，所有实例共享同一份
    #   实例变量：写在__init__里的 self.xxx，每个实例各有一份
    #   读取 self.block_size 时，实例上没有就会去类上找，所以实例方法里可以直接用 self.block_size
    # LLMEngine初始化时会用config.kvcache_block_size覆盖
    block_size = 256
    # 全局自增计数器，为每条序列分配唯一的seq_id
    # itertools.count()：无限的整数迭代器，每次 next(counter) 返回 0, 1, 2, ...
    # generate结束时按seq_id排序，使输出顺序与输入prompt顺序一致
    counter = count()

    # 注意：默认参数SamplingParams()只在函数定义时创建一次，所有未传参的调用共享同一个对象
    # 这里只读取它的字段、不会修改，所以没有问题
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 序列唯一编号，从0开始自增，也代表提交顺序
        self.seq_id = next(Sequence.counter)
        # 当前生命周期状态，新序列为WAITING
        self.status = SequenceStatus.WAITING
        # 全部token id：prompt + 已生成的token，生成过程中不断append
        # 浅拷贝，后续append_token不会影响调用方传进来的list
        #   如果直接 self.token_ids = token_ids，两个变量指向同一个list，append会改到调用方的数据
        #   copy() 创建一个新list，元素（int）相同，但list本身是独立的
        self.token_ids = copy(token_ids)
        # 最后一个token，decode时作为模型输入（只需输入它，前面的token都已在KV cache中）
        # [-1]：负数下标从末尾数，-1是最后一个元素
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
        # 例如：block_table=[5, 9, 12]，表示第0~255个token的KV在物理block 5，第256~511个在block 9，以此类推
        # 第j个token的KV所在槽位 slot = block_table[j // block_size] * block_size + j % block_size
        self.block_table = []
        # 采样参数直接拷贝到序列上，便于ModelRunner批量读取
        # 采样温度，见SamplingParams.temperature
        self.temperature = sampling_params.temperature
        # 最多生成多少个token（不含prompt），达到后序列结束
        self.max_tokens = sampling_params.max_tokens
        # 是否忽略EOS继续生成
        self.ignore_eos = sampling_params.ignore_eos

    # 以双下划线开头和结尾的方法叫"魔术方法"，Python在特定语法下自动调用
    # __len__：调用 len(seq) 时自动执行，返回总token数
    def __len__(self):
        return self.num_tokens

    # __getitem__：使用 seq[i] 或 seq[a:b] 时自动执行，key 为下标 i 或切片对象 slice(a, b)
    # 例如：seq[3] -> self.token_ids[3]；seq[0:5] -> self.token_ids[0:5]
    def __getitem__(self, key):
        return self.token_ids[key]

    # @property：装饰器，把方法变成"只读属性"，调用时不用加括号
    #   seq.is_finished    （正确，像访问字段一样）
    #   seq.is_finished()  （错误，返回的是bool，不能再调用）
    # 适合"由其他字段计算得到"的值，每次访问都重新计算，不会与其他字段不同步
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
    # //：整除（向下取整），7 // 2 = 3
    # 向上取整技巧：(a + b - 1) // b 等于 ceil(a / b)，避免用浮点数
    # 例如：block_size=256，num_tokens=600 -> (600 + 255) // 256 = 855 // 256 = 3个block
    #       num_tokens=512 -> (512 + 255) // 256 = 767 // 256 = 2个block（刚好填满，不多占）
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一个block里实际存了多少个token，取值范围[1, block_size]
    # = 总token数 - 前面(num_blocks-1)个满block的token数
    # 例如：block_size=256，num_tokens=600 -> 600 - 2*256 = 88
    #       num_tokens=512 -> 512 - 1*256 = 256（最后一个block刚好满）
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 取第i个逻辑block对应的token_ids，用于计算前缀缓存的hash
    # 切片 list[a:b] 取下标 [a, b) 的元素，b超出长度时自动截断，不会报错
    # 例如：block_size=256，num_tokens=600
    #   block(0) -> token_ids[0:256]，block(2) -> token_ids[512:768]，实际只有88个
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # decode每生成一个token就追加到序列末尾
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 自定义pickle序列化
    # pickle：Python内置的序列化库，把对象转成字节串（pickle.dumps），再还原（pickle.loads）
    #   默认会序列化对象的全部字段（self.__dict__）
    #   定义了__getstate__后，pickle只序列化它的返回值；还原时把这个值传给__setstate__
    #
    #   rank 0：pickle.dumps(seq) ──> 调用 seq.__getstate__() ──> 元组 ──> 字节串 ──> 共享内存
    #   rank 1：共享内存 ──> 字节串 ──> pickle.loads ──> 创建空对象 ──> 调用 __setstate__(元组)
    # 张量并行时，rank0需要通过共享内存把序列发送给其他rank，序列化的数据越小越好
    # 其他rank只负责模型前向，不做调度和采样，所以只发送前向需要的字段（不含seq_id、temperature等）
    # prefill：需要完整的token_ids来构造输入
    # decode：只需要最后一个token，避免每个step都传输整条序列，长序列时可省下大量序列化开销
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # 反序列化，与__getstate__对应
    # 通过last_state的类型判断发送方处于prefill（list）还是decode（int）
    # 注意：还原出来的对象不会执行__init__，所以只有这里设置过的字段才存在
    def __setstate__(self, state):
        # 元组解包：a, b, c = (1, 2, 3) 依次赋值给多个变量
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode阶段只拿到最后一个token，token_ids置空；
            # prepare_decode只用到last_token、num_tokens、block_table，不受影响
            self.token_ids = []
            self.last_token = last_state
