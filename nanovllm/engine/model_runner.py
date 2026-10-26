import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


# 模型执行器，每张GPU一个进程，执行一个ModelRunner实例
# 职责：加载模型、分配KV cache、把序列转为模型输入张量、执行前向、采样
# 张量并行时
#   rank0：主进程，由LLMEngine直接调用，同时通过共享内存给其他rank传输指令
#   rank1~N-1：子进程，ModelRunner构造完成后，进入loop()等待并执行rank0的指令
#
# 初始化流程（每个rank都执行一遍）
#   init_process_group：建立GPU间通信
#        │
#        ▼
#   load_model：加载本rank的那份权重
#        │
#        ▼
#   warmup_model：跑一次最大的prefill，测出激活值峰值显存
#        │
#        ▼
#   allocate_kv_cache：剩余显存全部分配给KV cache
#        │
#        ▼
#   capture_cudagraph：为decode录制CUDA Graph，enforce_eager时跳过
#        │
#        ▼
#   rank 0：创建共享内存，返回给LLMEngine
#   rank>0：连接共享内存，进入loop()等待命令
#
# 一次step的时序图（tp=2为例）
#      rank 0（主进程）              共享内存 / Event               rank 1（子进程）
#        │                                                             │ loop()中阻塞在
#        │                                                             │ event.wait()
#    call("run", seqs, True)                                           │
#        | write_shm：pickle序列化                                      │
#        | ["run", seqs, True] ───────> buf = [长度|字节串]              │
#        | event.set()───────────────> 亮灯 ───────────────────────────>│ 被唤醒
#        │                                                             │ read_shm：反序列化
#        │                                                             │ event.clear() 熄灯
#        ▼                                                             ▼
#   self.run(seqs, True)                                          self.run(seqs, True)
#        │ 每层的all_reduce <════════════ NCCL ════════════════════════>│
#        │ lm_head的gather  <══════════════════════════════════════════│ 发送本rank的logits
#        ▼                                                             ▼
#   采样，返回token_ids                                           返回None，继续loop()
# 
# 为什么不直接用NCCL传输序列？NCCL只能传GPU张量，序列是Python对象，用共享内存+pickle更简单
class ModelRunner:

    # config：全局配置
    # rank：进程在张量并行组内的编号，同时也是GPU编号
    # event：rank0传所有子进程的event（rank0通过所有子进程的event指挥子进程），其他rank传入自己的那个event
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        # HF模型配置（层数、hidden_size、头数、dtype等），从模型的config.json读取
        hf_config = config.hf_config
        # 每个kv block容纳的token数，用于计算slot编号
        self.block_size = config.kvcache_block_size
        # 是否不捕获CUDA Graph
        self.enforce_eager = config.enforce_eager
        # 张量并行的总进程数
        self.world_size = config.tensor_parallel_size
        # 张量并行组内的编号
        self.rank = rank
        # 进程间通知用的Event（rank 0：list；其他rank：单个）
        self.event = event

        # 初始化分布式进程组，进程通过rank编号标识，NCCL负责GPU之间通信
        # 所有rank都调用到这里后才会一起继续执行（类似集合点）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 每个rank绑定一张GPU
        torch.cuda.set_device(rank)
        # 临时修改默认dtype和device，之后创建的模型参数、KV cache、CUDA graph都以模型的dtype创建在GPU上
        # 省去.to()拷贝和类型转换
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 构建模型结构（参数为未初始化的空张量），再从safetensors文件加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # 采样器，logits -> token id
        self.sampler = Sampler()
        # 必须先预热，测出峰值激活显存，再用剩余的显存分配KV cache，最后捕获CUDA graph
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复默认设置，避免影响之后在CPU上的张量创建
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # rank 0创建1MB共享内存，用于向其他rank广播调用指令和参数
                # SharedMemory：多个进程可以同时读写一块内存，用name来标识内存空间
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # 所有rank都执行到barrier才往下走，保证rank0先创建，其他rank后连接
                dist.barrier()
            else:
                # 等待rank0创建完共享内存后，再连接
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 进入loop，直到收到exit才返回
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            # 等待所有rank都close之后，再由rank0删除共享内存
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            # 释放CUDA graph及内存池
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # 读取命令 -> 执行 -> 循环，直到收到exit
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # 非rank0从共享内存中读取命令
    # 内存布局
    #   buf[0:4]：数据长度，4字节小端整数
    #   buf[4:4+n]：pickle序列化后的[method_name, *args]
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        # 阻塞等待rank0写入
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 带*解包：第一个元素给method_name，剩下的全部作为列表给args
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 读完后清除事件，等待下一次通知
        self.event.clear()
        return method_name, args

    # rank 0 把指令写入共享内存，并通知所有子进程
    # 序列通过Sequence.__getstate__序列化，decode阶段只传最后一个token，减少传输量
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 统一的调用入口：rank 0先广播给其他rank，再在本地执行相同的方法
    # 所有rank都执行相同的方法，这样张量并行组内部的NCCL集合通信才能对齐
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 获取对象的属性/方法
        method = getattr(self, method_name, None)
        return method(*args)

    # 用最大规模的prefill跑一次前向
    #   测出峰值激活显存，计算kv可用显存
    #   触发torch.compile、cuBLAS等一次性开销
    def warmup_model(self):
        torch.cuda.empty_cache()
        # 重置峰值统计，确保统计到的是本次warmup的峰值
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 确保单次prefill单序列长度不超过max_model_len，计算的总token不超过max_num_batched_tokens
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        # 此时还没分配kv cache，所有token都需要计算，以此模拟最坏情况
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 用剩余显存分配KV cache，并把每层的KV cache视图挂到对应的attention模块上
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        # 整张卡已用显存 = 模型权重 + CUDA上下文 + 其他进程
        used = total - free
        # peak：warmup时峰值显存 
        # current：当前显存
        # peak - current = 前向激活时显存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 张量并行时，KV head被均分到各个rank
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 一个block占用字节数 = 2（K和V）* 层数 * 每个block容纳的token数 * kv头 * 每个kv头占用多少个dtype * 每个dtype占用多少个字节
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 计算GPU能放下多少个kv block
        # 可用与kv cache显存 = 总显存*利用率 - 已用显存 - 显存激活峰值
        #   整卡显存 total（例如80GB）
        #   ├── total × gpu_memory_utilization（72GB）：引擎允许使用的部分
        #   │     ├── used（2GB）：已经被占用的，主要是模型权重 + CUDA上下文
        #   │     ├── peak - current（3GB）：前向时激活值的临时峰值，用完即释放，但必须预留
        #   │     └── KV cache（72 - 2 - 3 = 67GB）：剩下的全部给KV cache
        #   └── total × (1 - gpu_memory_utilization)（8GB）：留给其他进程、显存碎片等
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 一整块连续的张量存放所有层的KV
        # kv_cache 就是一个普通的 torch.Tensor（默认device已设为cuda，所以直接在GPU上）
        # kv_cache: [2, 神经网络层数, block数量，每个block容纳多少个kv，每个kv有多少个头，每个kv头有多少维度]
        #   第0维：0为K，1为V
        #   第1维：层号
        #   第2维：物理block编号（即block_table中的值）
        #   第3维：block内的token偏移
        # 物理槽位 slot = block_id * block_size + 块内偏移，把[NB, BS]展开成长度为NB*BS的一维数组
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 给每一层分配对应的KV cache视图（不拷贝）
        # model.modules()：递归遍历模块中所有子模块（每层、每层的Linear等等）
        # hasattr：判断对象有没有这个属性，只有每一层的attention模块有k_cache/v_cache
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # k_cache/v_cache：[NB, BS, Hkv, D]
                # self.kv_cache[0, layer_id]：取某一层的所有k
                # 通过视图的方式，不拷贝，往视图写就是往kv_cache里写
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # 把各序列的block_table填充成等长二维张量，不足部分用-1补齐
    # 例如：seq0.block_table=[3, 7]，seq1.block_table=[5] -> [[3, 7], [5, -1]]
    # 为啥要补齐？
    #   1.张量必须是矩阵：torch.tensor([[3, 7], [5]])直接报错
    #   2.kernel按固定stride寻址：第b个序列直接用block_table[b * M + i]找到第i个逻辑块对应的物理块，整个batch一次launch就行，而不是每个序列各调用一次
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # pin_memory=True：在锁页内存中创建张量；普通内存可能会被操作系统换到磁盘，GPU不能直接读
        # .cuda(non_blocking=True)：异步拷贝到GPU，后续GPU计算会自动排在拷贝之后（CUDA stream），所以不会读到脏数据
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 构造prefill的模型输入：把一批序列本次要计算的token整理成计算所需的张量
    # 完整流程（block size=4，一个batch有A、B两条序列）
    #   1) 输入seqs，在CPU上，调度器已经分配好block，定好每条序列需要计算多少token
    #       A：a0~a5，6个token，num_cached_tokens=0，num_scheduled_token=6，block_table=[4, 1]
    #       B：b0~b6，7个token，num_cached_tokens=4，num_scheduled_token=3，block_table=[2, 5]
    #           （B的 b0~b3 命中前缀缓存，KV已经在 block 2 里，本次只算 b4~b6）
    #   2) 每条序列确定需要计算的区间[start, end) = [num_cached_token, num_cached_token + num_scheduled_token)
    #       A：[0, 6)   B：[4, 7)
    #   3) 逐条系列生成各个字段，再首尾拼接
    #     字段           A 的部分                  B 的部分        拼接后
    #     ──────────────────────────────────────────────────────────────────────────────────
    #     input_ids     a0 a1 a2 a3 a4 a5        b4 b5 b6       [a0 a1 a2 a3 a4 a5 b4 b5 b6]
    #     positions      0  1  2  3  4  5         4  5  6       [0 1 2 3 4 5 4 5 6]
    #     q长度               6                      3           [0, 6, 9]
    #     k长度               6                      7(含缓存)    [0, 6, 13]
    #     slot_mapping   16 17 18 19 4 5          20 21 22      [16 17 18 19 4 5 20 21 22]
    #     max_seqlen            max_seqlen_q = 6, max_seqlen_k = 7
    # 
    #     slot_mapping的计算（slot = 物理block编号 * block_size + 快内偏移）
    #       A：token 0~3 在逻辑block0 -> 在物理block4 -> slot 4*4+0 ~ 4*4+3 = 16 ~ 19
    #          token 4~5 在逻辑block1 -> 在物理block1 -> slot 1*4+0 ~ 1*4+1 = 4 ~ 5
    #       B：token 4~6 在逻辑block1 -> 在物理block5 -> slot 5*4+0 ~ 5*4+2 = 20 ~ 22
    #          token 0~3 在缓存里，不算也不写
    #   4) 判断是否需要 block_tables（命中前缀缓存，才需要block_tables记录偏移量，否则全部从0开始计算）
    #       cu_seqlens_k[-1]=13 > cu_seqlens_q[-1]=9，说明有缓存前缀，需要block_tables
    #   5) 拷贝所有list到CPU -> torch.tensor(pin_memory=True).cuda(non_blocking=True)
    #       input_ids、positions：作为模型输入参数
    #       cu_seqlens_q|k、max_seqlen_q|k、slot_mapping、block_tables：用于kv cache复用，非模型必要字段，放入全局上下文
    #   6) 这些字段在前向计算中的作用
    #        input_ids：embedding查表
    #        positions：RoPE，按位置旋转q、k
    #        slot_mapping：把token的KV写入kv_cache
    #        cu_seqlens_q/k + block_tables：作为flash_attn_varlen_func的参数
    #        cu_seqlens_q：获取每个序列最后一个token位置（lm_head只取每个序列最后一个token）
    #
    # 多条seq的token直接拼成一维，不做padding，用cu_seqlens记录每条序列的边界
    # flash_attn_varlen_func可直接处理这种拼接的数据，没有padding浪费
    #
    # 传统做法（padding）vs 拼接
    #   padding成[3, 5]：        拼接成[10]：
    #   [a a a _ _]             [a a a b b b b b c c]
    #   [b b b b b]              ↑     ↑       ↑   ↑
    #   [c c _ _ _]              0     3       8   10   <- cu_seqlens = [0, 3, 8, 10]
    #   浪费5个位置的计算           没有浪费；第i条序列 = 拼接张量[cu_seqlens[i] : cu_seqlens[i+1]]
    # 对应的pos(每个token在自己序列内的位置，用于RoPE）：
    #   [0 1 2 0 1 2 3 4 0 1]
    # 返回：input_ids [N], positions [N]
    def prepare_prefill(self, seqs: list[Sequence]):
        # 所有seq本次需要计算的token id拼接在一起
        input_ids = []
        # 每个token在自己序列中的位置，用于RoPE
        positions = []
        # 序列边界，形状是[B+1]
        # 第i条序列在拼接张量中的区间为[cu_seqlens[i], cu_seqlens[i+1]]
        # q：本次实际计算的token；k：参与注意力的全部token（缓存+本次计算）
        # 例如：两条序列，seq1计算3、缓存0；seq2计算100、缓存256
        #   cu_seqlens_q = [0, 3, 103]，cu_seqlens_k = [0, 3, 359]
        # 为啥q和k区间不一样？对于一条序列，第N个token的q，需要0~N个token的k来计算注意力
        # 为啥不需要cu_seqlens_v？k和v的缓存状态、长度和数组形状完全相同，再引入cu_seqlens_v很冗余
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        # batch内单条序列q/k长度的最大值
        # GPU的grid本身是矩形，每个线程计算一个token；为了让同一个batch所有seq的形状一致，只能按照最长序列的长度来规划
        max_seqlen_q = 0
        max_seqlen_k = 0
        # 每个token的kv写入kv cache的槽位，形状为[N]
        # 槽位 = block_id * block_size + block内偏移
        slot_mapping = []
        # 用于查找kv缓存，形状为[B, M]，M为batch内序列block_tables长度的最大值
        block_tables = None
        for seq in seqs:
            # 本次需要计算的token区间
            # start之前的token已有KV（缓存或之前的chunk），无需再计算
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            # k包含序列所有token
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 预热模式：禁用kv缓存，才能测出模型前向的显存激活峰值
            if not seq.block_table:    # warmup
                continue
            # 计算[start, end)每个token对应的slot，按照block逐段生成
            # 例如：block_size=256， block_table=[3,7,9], start=200, end=600
            #    start_block=0, end_block=3
            #    block0（物理block3）：[256 * 3 + 200: 256 * 3 + 256] = [968, 1024]
            #    block1（物理block7）：[256 * 7: 256 * 7 + 256] = [1792, 2048]
            #    block3（物理block9）：[9*256, 9*256+88) = [2304, 2392)
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                # 第一个block可能从中间开始（上一个chunk）
                if i == start_block:
                    slot_start += start % self.block_size
                # 中间block直接写满，最后一个block只写到end
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # 如果命中了KV缓存，则需要取出这部分KV缓存
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # list -> GPU tensor
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # attention相关元数据通过context传递给每层attention，而非层层传参
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 构造decode阶段的模型输入
    # 完整流程：block_size=4
    #   1) 输入：seqs
    #       A: a0~a6 (len=7)  last_token=a6   block_table=[4, 1]
    #       B: b0~b8 (len=9)  last_token=b8   block_table=[2, 5, 3]
    #            b8是第3个逻辑block的第1个token，may_append给B分配了block 3
    #   2) 取每条序列最后一个token，拼成长度为batch的列表
    #     字段           A              B              拼接后
    #     ─────────────────────────────────────────────────────────────
    #     input_ids     a6             b8             [a6, b8]
    #     positions     6              8              [6, 8]
    #     context_lens  7              9              [7, 9]
    #     slot_mapping  1*4+2=6        3*4+0=12       [6, 12]
    #     block_tables  [4, 1]         [2, 5, 3]      [[4, 1, -1]
    #                                                  [2, 5, 3]] 不够长的补-1
    #   3) list -> GPU tensor：同prefill，区别是decode不需要cu_seqlens，每条序列正好1个q，刚好对齐
    #   4) 这些字段在decode中的作用
    #      input_ids、positions：embedding查表；计算q、k；RoPE
    #      slot_mapping：每个token的KV写入位置
    #      block_tables+context_lens -> 每层 flash_attn_with_kvcache
    #                                     A：读block4的全部kv，读block1的前三个kv
    #                                     B：读block2和block5的全部kv，读block3的前1个kv
    #      lm_head：每条序列只有一个token，得到[2, V]
    # decode和prefill的区别
    #   prefill：每条序列是一段token，长度不一，用cu_seqlens标边界，可能没有block_table；返回input_ids [N]
    #   decode：每条序列只输入1个token，不需要cu_seqlens，一定有block_tables；返回input_ids [B]
    def prepare_decode(self, seqs: list[Sequence]):
        # 每条序列的最后一个token，[B]
        input_ids = []
        # 最后一个token在序列中的位置，[B]
        positions = []
        # 最后一个token的KV写入的槽位，[B]
        slot_mapping = []
        # 每条序列的长度，对于第i条序列，注意力需要读取context_lens[i]个KV，[B]
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 最后一个token的KV写到最后一个block的有效位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 采样参数，转为torch tensor，在GPU上进行采样
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 前向，返回每个词出现的概率logits：rank0为[batch, vocab_size]，其他rank为None
    # CUDA Graph是什么？
    #   eager：CPU逐个launch kernel，每个kernel都有几微秒的启动开销
    #      CPU：[launch matmul][launch add][launch norm]   一次前向成百上千个kernel
    #      GPU：     [matmul]    [add]      [norm]         kernel很小时，GPU经常等待CPU
    #   CUDA Graph：提前把整个kernel序列录制成一张图，之后一条命令重放整张图
    #      CPU：[replay]
    #      GPU：[matmul][add][norm]                        GPU连续执行，没有空隙
    # decode每条序列只算1个token，kernel都很小，启动开销占比高，CUDA Graph收益大
    # 代价：每个kernel读写的显存地址是录制时固定的，所以输入必须拷到固定的缓冲区
    # CUDA Graph vs torch.compile
    #   CUDA Graph：减少kernel启动次数
    #   torch.compile：合并代码，减少访存次数，c = a+b, e = c + d -> e = a+b+d
    # inference_mode：比no_grad（只关闭建图）更彻底，关闭版本计数和view追踪（追踪in-place、view操作）
    # @装饰器：写在函数定义上方，等价于 run_model = torch.inference_mode()(run_model)，给函数加上额外行为
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 以下情况强制走eager模式
        #   prefill：token数变化大，无法为每个形状都捕获CUDA graph；大kernel，计算密集型，kernel启动开销占比小
        #   enforce_eager：用户强制关闭CUDA Graph
        #   batch > x：超过捕获的最大batch
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # model(): [B, 1] -> [B, H]
            # compute_logits(): [B, H] -> [B, V]
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 走CUDA Graph：重放提前录制好的kernel序列，省去成百上千次kernel launch的开销
            bs = input_ids.size(0)
            context = get_context()
            # 选择不小于实际batch的最小graph，多出来的用padding补齐
            # 例如：bs=21，使用bs=32的graph，后11行是padding
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # CUDA Graph的输入输出地址在捕获时就固定了，只能把数据拷贝到这些固定缓冲区中
            # 张量[:bs] = 另一个张量：把数据拷贝到前bs个位置，原地写入，地址不变
            # 注意不能用graph_vars["input_ids"] = input_ids，那样只是把字典指向新张量，graph读不到
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # fill_(-1)：所有元素原地赋值为-1；PyTorch中以下划线结尾的方法都是"原地操作"，直接修改自身
            # padding部分下标填充-1，表示跳过此位置
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # padding部分长度置0，表示不读取KV
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # [max_bs, max_num_blocks] 的 左上角 [bs, M] 区域
            # 例如，graph是按照max_num_blocks申请的缓冲区（假设为5），本batch序列最大block数是3，每条序列最后两位补0
            #    2  43  54  0  0
            #    4   5   9  0  0
            #    2   1  31  0  0
            #    0   0   0  0  0 [补一行]
            #               [补0]
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放录制好的kernel序列：从缓冲区读输入，结果写到固定的ouputs缓冲区
            graph.replay()
            # [max_bs, H] -> [bs, H] -> [bs, V]
            # 为啥compute_logits不录入graph
            #   1. 输出在各rank不对称：rank 0 返回 [bs, V]，其他 rank 返回 None；graph 要求每个 rank 都有固定形状、固定地址的输出 buffer
            #   2. graph边界停在hidden state更干净：把 graph 的输出定在 outputs [max_bs, H]，graph 只依赖模型主体；每个 batch 的采样配置都可能不同，放在 graph 外可以随意改
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 一次run()完整数据流：prefill，2条序列A、B，本次分别计算3个、2个token -> bs=2, N=5
    #    seqs = [A, B]                 CPU上的python对象
    #      │
    #      │ prepare_prefill：python列表 -> torch tensor
    #      │---> input_ids [5], positions [5]   作为参数传给模型
    #      │---> cu_seqlens_q/k [3], slot_mapping [5]，block_tables [2, M] 放入全局Context
    #      │
    #      │ prepare_sample：只在rank0
    #      │---> temperatures [2]
    #      ▼
    #    run_model：
    #      prefill ---> self.model(...)，逐个启动kernel
    #      decode  ---> 把输入拷贝到CUDA Graph缓冲区，graph.replay()
    #      │
    #      │---> hidden_states [5, H]           prefill会为每个token计算hidden state
    #      │      │ compute_logits：取每条序列最后一个token
    #      │      ▼
    #      └──> logits [2, V]    每条序列在词表上的打分
    #      │
    #      ▼
    #    sampler -> token_ids [2] GPU张量
    #      │ .tolist(): 拷贝回cpu
    #      ▼
    #   [15, 8]（Python列表）──> 返回给 LLMEngine.step ──> scheduler.postprocess 追加到各序列
    #
    # 最后，reset_context清空元数据，避免下一步被误用
    # 其他rank执行相同的run()，但不采样，返回None
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 采样只在rank0进行
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # 为decode阶段捕获不同batch大小的CUDA Graph
    # CUDA Graph会把一次前向所有kernel的启动参数录制下来，之后replay只需要一次启动
    # 要求：输入输出张量地址固定、形状固定；所以要预先分配静态缓冲区，并为每种batch大小单独捕获
    # 1) 核心原理：graph记住的是“地址”，每次decode只换地址里的数据
    #     显存：静态缓冲区（地址固定）                graph：kernel+地址
    #     ┌────────────────────────────┐         ┌───────────────────────────┐
    #     │ 0x100  input_ids           │<── 读 ── │ kernel 1：读0x100，写0x500 │
    #     │ 0x300  positions、slot_map │<──读/写 ─ │ kernel 2：读0x300，写0x500 │
    #     │ 0x500  中间结果             │ <──────  │ ...（28层，几百个kernel）   │      
    #     │ ...                        │          │                           │
    #     │ 0x900  outputs             │<── 写 ── │ kernel n：读 ...，写 0x900 │
    #     └────────────────────────────┘         └───────────────────────────┘
    #     第1次decode： 0x100 <- [a6, b8]   ──> replay ──>  0x900 中得到 a6、b8 的输出
    #     第2次decode： 0x100 <- [a7, b9]   ──> replay ──>  0x900 中得到 a7、b9 的输出
    #                    ↑                       ↑
    #                    只换缓冲区里的数据      同一张graph：同样的kernel、同样的地址
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        # 单条序列最多需要的block数
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 静态缓冲区，所有graph共用（按照[:bs]切片）
        # [max_bs]
        # decode阶段输入数据大部分是一维，形状相对固定
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        # [max_bs, max_num_blocks]
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        # [max_bs, hidden_size]
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 捕获的batch大小，1,2,4,8各一档，之后每隔16一档，直到max_bs
        # 细分小batch，减少padding浪费；粗分大batch，减少捕获时间和显存
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        # 所有graph共享一个显存池
        self.graph_pool = None

        # 从大到小捕获，第一个graph申请的内存池最大，后续小graph共享这个内存池
        #      graph_pool：|<─────────────── graph[512] 的中间结果 ───────────────>|
        #                  |<────── graph[256] ──────>|
        #                  |<─ graph[1] ─>|
        # 同一时刻只会重放一个graph，所以大家共用同一块显存不会冲突
        for bs in reversed(self.graph_bs):
            # 创建一个空的CUDA Graph，此时里面什么都没有，相当于一盘空录像带
            graph = torch.cuda.CUDAGraph()
            # 把注意力需要的元数据放进全局上下文
            # 传入的是切片，地址本身就是缓冲区的地址
            # 所以录制阶段录下的kernel读的正是这几块缓冲区，以后replay前直接改缓冲区内容即可
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 先跑一遍，触发torch.compile、cuBLAS等初始化，这些操作不能出现在graph里
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # 录制：with块内的GPU操作不会真正执行，而是按顺序记入graph
            # 录下的内容包括
            #   - 前向所有kernel（embedding、n层的matmul/RoPE/写KV/attention/MLP
            #   - 每个kernel读写显存的地址
            #   - 最后把结果写入outputs的拷贝操作
            # self.graph_pool第一轮为None，表示新建一个内存池
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                 # 把刚建的池保存下来，之后不再是 None
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            # 等待GPU完成本轮所有工作，再进入下一轮
            torch.cuda.synchronize()
            # 清空全局上下文，避免下一轮或之后误用这些元数据
            reset_context()

        # 保存缓冲区，run_model往里面填数据，然后replay
        #   run_model 需要通过它们找到缓冲区，往里面拷贝decode所需数据，从output取结果
        #   需要保持对这些张量的引用：graph只是记住了地址，并不持有这些张量；如果函数结束后它们被python回收，显存被释放，replay就会读到已释放的显存
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
