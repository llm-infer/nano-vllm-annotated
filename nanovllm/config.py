import os
from dataclasses import dataclass
from transformers import AutoConfig


#====================================================================================================
# 全局视角：Config 在引擎中的流转，以及每个字段被谁使用
#====================================================================================================
#
# 创建与填充的时间线（同一个Config对象被各组件共享，后面的修改前面的组件也能看到）：
#
#   LLM(path, **kwargs)
#     │ LLMEngine.__init__：从kwargs中挑出Config的字段，创建 Config
#     │   __post_init__：校验参数，读取 hf_config，修正 max_model_len
#     ▼
#   ModelRunner(config, rank, ...)
#     │ allocate_kv_cache：根据剩余显存算出并写入 config.num_kvcache_blocks（原来是-1）
#     ▼
#   LLMEngine：从tokenizer读取并写入 config.eos（原来是-1）
#     ▼
#   Scheduler(config)：此时 num_kvcache_blocks 和 eos 都已就绪，创建 BlockManager
#
#   注意：张量并行时，子进程（rank 1~N-1）拿到的是Config经过pickle复制的副本，
#   rank 0 上之后的修改（如eos）不会同步过去；子进程会在自己的 allocate_kv_cache 中各自算出 num_kvcache_blocks
#
# 字段 -> 使用者：
#
#   字段                        谁使用                                 用来做什么
#   ────────────────────────────────────────────────────────────────────────────────────────────────
#   model                       load_model、AutoTokenizer              读取权重和分词器
#   max_num_batched_tokens      Scheduler                              每步prefill的token预算
#                               ModelRunner.warmup_model               决定预热batch的规模
#   max_num_seqs                Scheduler                              每步最多调度多少条序列
#                               ModelRunner.capture_cudagraph          最大捕获batch（不超过512）
#   max_model_len               ModelRunner.warmup_model               预热序列长度
#                               ModelRunner.capture_cudagraph          block_tables缓冲区的列数
#   gpu_memory_utilization      ModelRunner.allocate_kv_cache          计算KV cache可用显存
#   tensor_parallel_size        LLMEngine                              启动几个子进程
#                               ModelRunner                            进程组大小、kv头切分
#   enforce_eager               ModelRunner                            是否捕获/使用CUDA Graph
#   hf_config                   ModelRunner、Qwen3ForCausalLM          构建模型、计算block字节数
#   eos                         Scheduler.postprocess                  判断序列是否结束
#   kvcache_block_size          Sequence、BlockManager                 按block切分token、分配block
#                               ModelRunner                            计算slot、kv_cache形状
#   num_kvcache_blocks          BlockManager                           物理block总数
#====================================================================================================


# dataclass：根据类里的字段声明，自动帮你生成样板方法，不用再手写 __init__、__repr__、__eq__
# slots=True：
#   作用：为每个字段预留一个固定位置，而不是用__dict__存放字段；
#   好处：省内存，一个空字典本身就要占几十到上百字节，而且字典为了查找效率会预留空间。槽位只存指针，没有这层开销
# 其他常用选项：
#   frozen=True：实例不可变
#   eq=False：不生成 __eq__
@dataclass(slots=True)
class Config:
    # 模型所在的本地目录路径
    model: str
    # 单个step允许处理的最大token
    # 主要约束prefill：把多个prompt token请求拼接到一个batch，token总和不能大于它；decode每个seq只处理一个token，不会触发上限约束
    # 调大： 
    #   吞吐更高：权重读取（每次计算qkv，都要读一遍权重）、固定开销（kernel launch、h2d、d2h）被摊薄
    #   单步更慢：PD一体计算，ITL会变大，新请求等待上一个大prefill step，TTFT会变大，缓解办法是 chunked prefill（把长 prompt 切成块，和 decode 混在同一步里）
    #   激活显存更大：导致可用kv block变小，能同时服务的在线请求变少；容易导致OOM
    max_num_batched_tokens: int = 16384
    # 单个step允许处理的序列数上限
    # decode是访存受限，每一个step基本由"读权重"决定，和 batch 里有几条序列几乎无关，所以 batch 里多放一条序列，几乎是白赚的一个 token
    # 但batch也存在上限，实际 batch 受限显存大小， = min(max_num_seqs, 当前在线请求数, KV cache 能容纳的序列数)
    max_num_seqs: int = 512
    # 单序列的最大上下文长度（prompt+生成的token之和），超过即截断
    # 不能超过模型自身存在最大上限
    max_model_len: int = 4096
    # 允许引擎使用的 GPU 显存比例(0~1)
    # 总显存 × gpu_memory_utilization = 模型权重 + 峰值激活 + KV cache
    # 总显存 × (1 - gpu_memory_utilization) = CUDA 上下文、碎片、其他进程
    gpu_memory_utilization: float = 0.9
    # 把每层权重切到多张卡上
    tensor_parallel_size: int = 1
    # True = 强制 eager 模式, 不使用 CUDA Graph。
    # CUDA Graph 把多个kernel录制到一个流水线上，能显著降低 decode 阶段的 kernel 启动开销
    enforce_eager: bool = False
    # 模型配置对象(层数、head 数、隐藏维度、最大上下文等)
    hf_config: AutoConfig | None = None
    # 结束符，遇到这个token就停止生成
    # -1表示未设置
    # 通常从tokenizer.eos_token_id中读取
    eos: int = -1
    # 一个kv block能容纳多少个 token
    # block不必再显存中连续，减少显存碎片，支持前缀复用计算
    # 值越大，管理的block数量越少，但资源浪费越大（最后一个block可能填不满）
    kvcache_block_size: int = 256
    # kv block的数量
    # 一般不指定，由引擎通过剩余显存计算出来
    num_kvcache_blocks: int = -1

    # 钩子函数，__init__执行完后自动执行
    # 通常用来校验参数和初始化派生字段
    def __post_init__(self):
        # 校验模型目录是否存在
        assert os.path.isdir(self.model)
        # paged attention对kv block大小有要求，必须是256的倍数
        assert self.kvcache_block_size % 256 == 0
        # 限制只能在单机下运行
        assert 1 <= self.tensor_parallel_size <= 8
        # 从本地模型目录读取 config.json，填入hf_config
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 把max_model_len限制在模型支持的最大长度以内
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
