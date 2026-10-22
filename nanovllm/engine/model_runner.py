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


# 模型执行器，每张GPU（每个TP rank）一个实例
# 职责：加载模型、分配KV cache、捕获CUDA Graph、把序列转成模型输入张量、执行前向和采样
# 张量并行时：
#   rank 0 在主进程中，由LLMEngine直接调用，同时通过共享内存把调用广播给其他rank
#   rank 1~N-1 在子进程中，构造完成后进入loop()，等待并执行rank 0广播的调用
#
# 本文件中的shape记号：
#   B  ：本step的序列数（batch size）
#   N  ：本step所有序列拼接后的token总数（prefill时为各序列num_scheduled_tokens之和，decode时N=B）
#   L  ：模型层数 num_hidden_layers
#   NB ：KV cache的block总数 num_kvcache_blocks
#   BS ：每个block容纳的token数 block_size
#   Hkv：当前rank上的kv头数 num_key_value_heads / tp_size
#   D  ：每个头的维度 head_dim
#   H  ：隐藏层维度 hidden_size
#   V  ：词表大小 vocab_size
#   M  ：block_table的列数（batch中block数最多的序列的block数）
#
# 初始化流程（每个rank都执行一遍）：
#
#   init_process_group（建立GPU间通信）
#        │
#        ▼
#   构建模型 + load_model（加载本rank那一份权重）
#        │
#        ▼
#   warmup_model（跑一次最大的prefill，测出激活值峰值显存）
#        │
#        ▼
#   allocate_kv_cache（剩余显存全部分给KV cache）
#        │
#        ▼
#   capture_cudagraph（为decode录制CUDA Graph，enforce_eager时跳过）
#        │
#        ▼
#   rank 0：创建共享内存，返回给LLMEngine使用
#   rank>0：连接共享内存，进入loop()等待指令
#
# 张量并行时rank 0与其他rank的交互（时序图，以一次 call("run", seqs, True) 为例）：
#
#      rank 0（主进程）              共享内存 / Event               rank 1（子进程）
#        │                                                               │ loop()中阻塞在
#        │                                                               │ event.wait()
#   call("run", seqs, True)                                              │
#        │ write_shm：pickle序列化                                       │
#        │ ["run", seqs, True] ───────> buf = [长度|字节串]              │
#        │ event.set() ───────────────> 亮灯 ───────────────────────────>│ 被唤醒
#        │                                                               │ read_shm：反序列化
#        │                                                               │ event.clear() 熄灯
#        ▼                                                               ▼
#   self.run(seqs, True)                                            self.run(seqs, True)
#        │ 每层的all_reduce <════════════ NCCL ═════════════════════════>│
#        │ lm_head的gather  <════════════════════════════════════════════│ 发送本rank的logits
#        ▼                                                               ▼
#   采样，返回token_ids                                             返回None，继续loop()
#
# 为什么不直接用NCCL传输序列？NCCL只能传GPU张量，序列是Python对象，用共享内存+pickle更简单
class ModelRunner:

    # config：全局配置
    # rank：当前进程在张量并行组中的编号，同时也是使用的GPU编号
    # event：rank 0传入所有子进程的Event列表；其他rank传入自己的那一个Event
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 全局配置，allocate_kv_cache会回写config.num_kvcache_blocks
        self.config = config
        # HF模型配置（层数、头数、hidden_size、dtype等）
        hf_config = config.hf_config
        # 每个KV block容纳的token数，用于计算slot（槽位）编号
        self.block_size = config.kvcache_block_size
        # True则不捕获CUDA Graph，所有前向都走eager
        self.enforce_eager = config.enforce_eager
        # 张量并行的总进程数（GPU数）
        self.world_size = config.tensor_parallel_size
        # 当前进程的编号
        self.rank = rank
        # 进程间通知用的Event（rank 0：list；其他rank：单个）
        self.event = event

        # 初始化分布式进程组，NCCL负责GPU之间的通信（all_reduce、gather）
        # 所有rank都调用到这里后才会一起继续执行（类似集合点）
        # tcp://localhost:2333：rendezvous地址，各rank通过它互相发现，只支持单机
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 每个rank绑定一张GPU
        torch.cuda.set_device(rank)
        # 临时修改默认dtype和device，之后创建的模型参数、KV cache、CUDA Graph缓冲区
        # 都直接以模型的dtype（如bfloat16）创建在GPU上，省去.to()的拷贝
        #   例如 torch.zeros(3) 默认是 float32、CPU；设置后变为 bfloat16、GPU
        #   bfloat16：16位浮点数，比float32省一半显存，精度略低但对推理足够
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 构建模型结构（参数为未初始化的空张量），再从safetensors加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # 采样器：logits -> token id
        self.sampler = Sampler()
        # 顺序很重要：先预热测出峰值激活显存，再用剩余显存分配KV cache，最后捕获CUDA Graph
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复默认设置，避免影响之后CPU上的张量创建
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # rank 0创建1MB共享内存，用于向其他rank广播调用指令和参数
                # SharedMemory：多个进程可以同时读写的一块内存，用name来找到同一块
                # 2**20 = 1048576 字节 = 1MB
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # barrier：所有rank都执行到barrier后才一起往下走，保证rank 0先创建、其他rank后连接
                dist.barrier()
            else:
                # 等rank 0创建完共享内存后再连接
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 子进程进入事件循环，直到收到exit才返回
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            # 等所有rank都close之后，再由rank 0删除共享内存
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            # 释放CUDA Graph及其内存池
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # 非0 rank的事件循环：读取指令 -> 执行 -> 循环，收到exit时退出
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # 非0 rank从共享内存读取指令
    # 内存布局：
    #   buf[0:4]    ：数据长度n，4字节小端整数
    #   buf[4:4+n]  ：pickle序列化后的 [method_name, *args]，例如 ["run", seqs, True]
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        # 阻塞等待rank 0写入完成
        self.event.wait()
        # int.from_bytes：4个字节 -> 整数（与write_shm中的to_bytes相反）
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 带*的解包：第一个元素给method_name，剩下的全部作为列表给args
        #   例如 a, *b = ["run", seqs, True] -> a="run"，b=[seqs, True]
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 读完后清除事件，等待下一次通知
        self.event.clear()
        return method_name, args

    # rank 0把指令写入共享内存，并通知所有子进程
    # 序列通过Sequence.__getstate__序列化，decode阶段只传最后一个token，数据量很小
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # [method_name, *args]：*args在列表中展开，例如 ["run", seqs, True]
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 整数 -> 4个字节（小端序：低位字节在前），例如 300 -> b'\x2c\x01\x00\x00'
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 统一的调用入口：rank 0先广播给其他rank，再在本地执行同名方法
    # 所有rank执行相同的方法，模型内部的NCCL集合通信（all_reduce等）才能对齐
    # *args：可变位置参数，调用时多传的参数都收集成一个元组
    #   call("run", seqs, True) -> method_name="run"，args=(seqs, True)
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # getattr(对象, "名字")：按字符串名字取对象的属性/方法
        #   getattr(self, "run") 等价于 self.run
        # 这样可以把"要调用哪个方法"作为字符串在进程间传递
        method = getattr(self, method_name, None)
        # method(*args)：把元组展开成位置参数，等价于 self.run(seqs, True)
        return method(*args)

    # 用最大规模的prefill跑一次前向，目的：
    #   1. 测出前向过程中的峰值激活显存，供allocate_kv_cache计算可用显存
    #   2. 触发torch.compile编译、cuBLAS初始化等一次性开销
    def warmup_model(self):
        torch.cuda.empty_cache()
        # 重置峰值统计，确保测到的是本次前向的峰值
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 构造token总数接近max_num_batched_tokens的batch，模拟最坏情况
        # 例如：max_num_batched_tokens=16384，max_model_len=4096 -> 4条长度为4096的序列，N=16384
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        # 此时还没分配KV cache，序列也没有block_table，prepare_prefill和Attention都会跳过写cache
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 用剩余显存分配KV cache，并把每层的K/V cache视图挂到对应的Attention模块上
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # free/total：整张卡的空闲/总显存（字节），包括其他进程、CUDA上下文等非PyTorch占用
        free, total = torch.cuda.mem_get_info()
        # 当前整卡已用显存 = 模型权重 + CUDA上下文 + PyTorch缓存 + 其他进程
        used = total - free
        # peak：预热时PyTorch分配的峰值；current：当前PyTorch分配量（主要是模型权重）
        # peak - current 即前向时激活值临时占用的显存，需要为它预留空间
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 张量并行时，KV头被均分到各个rank
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 一个block占用的字节数 = 2(K和V) × L × BS × Hkv × D × dtype字节数
        # 例如Qwen3-0.6B（28层，8个kv头，head_dim=128，bf16）：2×28×256×8×128×2 ≈ 28MB
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 可用于KV cache的显存 = 总显存×利用率 - 已用显存 - 激活峰值增量
        #
        #   整卡显存 total（例如80GB）
        #   ├── total × gpu_memory_utilization（72GB）：引擎允许使用的部分
        #   │     ├── used（2GB）：已经被占用的，主要是模型权重 + CUDA上下文
        #   │     ├── peak - current（3GB）：前向时激活值的临时峰值，用完即释放，但必须预留
        #   │     └── KV cache（72 - 2 - 3 = 67GB）：剩下的全部给KV cache
        #   └── total × (1 - gpu_memory_utilization)（8GB）：留给其他进程、显存碎片等
        #
        # 例如：80GB卡，利用率0.9，已用2GB，激活峰值增量3GB -> (72-2-3)GB / 28MB ≈ 2450个block ≈ 62万token
        # int(...)：转为整数；// block_bytes：能放下多少个完整的block
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 一整块连续张量存放所有层的KV
        # kv_cache: [2, L, NB, BS, Hkv, D]
        #   第0维：0为K，1为V
        #   第1维：层号
        #   第2维：物理block编号（即block_table中的值）
        #   第3维：block内的token偏移
        # 物理槽位 slot = block_id × BS + 块内偏移，把[NB, BS]展平后即为[NB*BS]个槽位
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 按模块注册顺序（即层的顺序）给每个Attention模块分配对应层的K/V cache视图（不拷贝）
        # model.modules()：递归遍历模型中的所有子模块（包括模型自己、每层、每层里的Linear等）
        # hasattr(对象, "名字")：判断对象有没有这个属性，只有Attention模块有k_cache/v_cache
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # k_cache / v_cache: [NB, BS, Hkv, D]
                # self.kv_cache[0, layer_id]：前两维各取一个下标，剩下的维度保留
                #   [2, L, NB, BS, Hkv, D] -> [NB, BS, Hkv, D]
                # 这是"视图"（view）：和kv_cache共用同一块显存，没有拷贝；往视图里写数据就是往kv_cache里写
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # 把各序列的block_table填充成等长二维张量，不足的部分用-1补齐
    # 返回 block_tables: [B, M]，M = 最长的block_table长度
    # 例如：seq0.block_table=[3, 7]，seq1.block_table=[5] -> [[3, 7], [5, -1]]
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # torch.tensor(Python列表, dtype=...)：把Python的嵌套列表转成张量，[[3, 7], [5, -1]] -> 形状[2, 2]
        # pin_memory=True：在"锁页内存"中创建张量；普通内存可能被操作系统换到磁盘，GPU不能直接读，
        #   锁页内存保证常驻物理内存，GPU可以用DMA直接读取，拷贝更快，而且才能做异步拷贝
        # .cuda(non_blocking=True)：拷贝到GPU；non_blocking表示CPU发出拷贝命令后不等它完成就继续往下执行，
        #   后续GPU上的计算会自动排在拷贝之后，所以不会读到不完整的数据
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 构造prefill的模型输入
    # 多条序列的token直接拼接成一维（不做padding），用cu_seqlens记录每条序列的边界，
    # 供flash_attn_varlen_func使用，没有padding浪费
    #
    # 传统做法（padding）vs 这里的做法（拼接），3条序列长度分别为 3、5、2：
    #
    #   padding成[3, 5]：           拼接成[10]：
    #   [a a a _ _]                 [a a a b b b b b c c]
    #   [b b b b b]                  ↑     ↑         ↑   ↑
    #   [c c _ _ _]                  0     3         8   10   <- cu_seqlens = [0, 3, 8, 10]
    #   浪费5个位置的计算             没有浪费；第i条序列 = 拼接张量[cu_seqlens[i] : cu_seqlens[i+1]]
    #
    # 对应的positions（每个token在自己序列内的位置，用于RoPE）：
    #   [0 1 2 0 1 2 3 4 0 1]
    # 返回：input_ids [N]，positions [N]
    def prepare_prefill(self, seqs: list[Sequence]):
        # 所有序列本次要计算的token id拼接在一起，最终[N]
        input_ids = []
        # 每个token在自己序列中的位置（用于RoPE），最终[N]
        positions = []
        # 累积序列长度（cumulative sequence lengths），最终[B+1]
        # 第i条序列在拼接张量中的区间为[cu_seqlens[i], cu_seqlens[i+1])
        # q：本次实际计算的token；k：参与注意力的全部token（缓存的前缀 + 本次计算的）
        # 例如：两条序列，seq1缓存0个、计算3个；seq2缓存256个、计算100个
        #   cu_seqlens_q = [0, 3, 103]，cu_seqlens_k = [0, 3, 359]
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        # batch内单条序列q/k长度的最大值，标量
        max_seqlen_q = 0
        max_seqlen_k = 0
        # 每个新token的KV要写入的cache槽位，最终[N]
        # slot = 物理block_id × block_size + block内偏移
        slot_mapping = []
        # 仅在存在缓存前缀时才需要，[B, M]
        block_tables = None
        for seq in seqs:
            # 本次计算的token区间[start, end)，start之前的token已有KV（前缀缓存或之前的chunk）
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            # k包含从序列开头到end的所有token
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # 计算[start, end)每个token对应的slot，按block逐段生成
            # 例如：BS=256，block_table=[3, 7, 9]，start=200，end=600
            #   start_block=0，end_block=3
            #   block 0（物理3）：slot [3*256+200, 3*256+256) = [968, 1024)
            #   block 1（物理7）：slot [7*256, 8*256)        = [1792, 2048)
            #   block 2（物理9）：slot [9*256, 9*256+88)     = [2304, 2392)
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                # 第一个block可能从中间开始（chunked prefill的后续chunk）
                if i == start_block:
                    slot_start += start % self.block_size
                # 中间的block写满，最后一个block只写到end为止
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # k比q长，说明有序列存在已缓存的前缀（前缀缓存命中或chunked prefill），
        # 注意力需要通过block_table从KV cache中读取这部分KV
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # list -> GPU张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)          # [N]
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)          # [N]
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)    # [B+1]
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)    # [B+1]
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)    # [N]（warmup时为[0]）
        # 注意力相关的元数据通过全局context传递给每层的Attention，而不是层层传参
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 构造decode的模型输入：每条序列只输入最后一个token，N = B
    # 返回：input_ids [B]，positions [B]
    def prepare_decode(self, seqs: list[Sequence]):
        # 每条序列的最后一个token，[B]
        input_ids = []
        # 最后一个token的位置 = len - 1，[B]
        positions = []
        # 最后一个token的KV写入的槽位，[B]
        slot_mapping = []
        # 每条序列的上下文长度（含本次输入的token），注意力需要读取这么多个位置的KV，[B]
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 最后一个token的KV写到最后一个block的最后一个有效位置
            # 例如：BS=256，len=600，block_table[-1]=9 -> last_block_num_tokens=88，slot=9*256+87
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)          # [B]
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)          # [B]
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)    # [B]
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)    # [B]
        block_tables = self.prepare_block_tables(seqs)                                                           # [B, M]
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 采样参数，目前只有temperature，每条序列可以不同
    # 返回 temperatures: [B]，float32
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 前向，返回logits：rank 0为[B, V]，其他rank为None
    #
    # CUDA Graph是什么？
    #   普通（eager）执行：CPU逐个发射kernel，每个kernel都有几微秒的启动开销
    #     CPU: [发射matmul][发射add][发射norm][发射matmul]...  一次前向几百~上千个kernel
    #     GPU:     [matmul]  [add]   [norm]   [matmul]...      kernel很小时，GPU经常在等CPU
    #   CUDA Graph：提前把整个kernel序列录制成一张"图"，之后一条命令重放整张图
    #     CPU: [replay]
    #     GPU: [matmul][add][norm][matmul]...                  GPU连续执行，没有空隙
    #   decode时每条序列只算1个token，kernel都很小，启动开销占比高，用CUDA Graph收益很大
    #   代价：图里每个kernel读写的显存地址是录制时固定的，所以输入必须拷到固定的缓冲区里
    # 注意prefill时lm_head只取每条序列的最后一个token，所以输出也是[B, V]而不是[N, V]
    # inference_mode：比no_grad更彻底，关闭梯度追踪和版本计数，推理更快
    #   训练时PyTorch会记录每一步计算（计算图），用于反向传播求梯度；推理不需要，关掉可以省显存、提速
    # @装饰器：写在函数定义上方，等价于 run_model = torch.inference_mode()(run_model)，给函数加上额外行为
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 以下情况走eager模式：
        #   prefill：token数变化范围太大，无法为每种形状都捕获CUDA Graph，且prefill是计算密集型，kernel启动开销占比小
        #   enforce_eager：用户强制关闭CUDA Graph
        #   batch > 512：超过捕获的最大batch
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # model(...)：[N] -> hidden_states [N, H]；compute_logits：[N, H] -> [B, V]
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # decode走CUDA Graph：重放预先录制好的kernel序列，省去几百次kernel launch的CPU开销
            bs = input_ids.size(0)
            context = get_context()
            # 选择不小于实际batch的最小已捕获graph，多出来的位置当作padding
            # 例如：bs=21 -> 使用bs=32的graph，后11行是padding
            # next(生成器)：取生成器的第一个元素，即graph_bs中第一个 >= bs 的值
            #   graph_bs = [1, 2, 4, 8, 16, 32, ...]，bs=21 -> 32
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # CUDA Graph的输入输出地址在捕获时就固定了，只能把数据拷贝到这些固定缓冲区中
            # 张量[:bs] = 另一个张量：把数据拷贝进前bs个位置（原地写入，地址不变）
            # 注意不能写成 graph_vars["input_ids"] = input_ids，那样只是让字典指向新张量，graph读不到
            graph_vars["input_ids"][:bs] = input_ids          # [max_bs] 的前bs个
            graph_vars["positions"][:bs] = positions          # [max_bs] 的前bs个
            # padding部分slot置为-1，store_kvcache会跳过，不会写坏KV cache
            # fill_(-1)：所有元素原地赋值为-1；PyTorch中以下划线结尾的方法都是"原地操作"，直接修改自身
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # padding部分上下文长度置0，注意力不读取任何KV
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # [max_bs, max_num_blocks] 的左上角 [bs, M] 区域
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放录制好的kernel序列：从固定缓冲区读输入，结果写到固定的outputs缓冲区
            graph.replay()
            # outputs: [max_bs, H] -> 取前bs行 [bs, H] -> lm_head -> [bs, V]
            # lm_head不在graph内（输出到rank 0需要gather）
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 一次 run() 的完整数据流（例：prefill，2条序列A、B，本次分别计算3个、2个token -> B=2，N=5）：
    #
    #   seqs = [A, B]                                    CPU上的Python对象（Sequence）
    #      │
    #      │ prepare_prefill（decode时为prepare_decode）：Python列表 -> 张量，.cuda()拷到GPU
    #      ├──> input_ids [5]、positions [5]             作为参数传给模型
    #      ├──> cu_seqlens_q/k [3]、slot_mapping [5]、   set_context 放进全局 _CONTEXT，
    #      │    block_tables [2, M]（有前缀时）          每一层 Attention 用 get_context() 读取
    #      │
    #      │ prepare_sample（只在rank 0）
    #      ├──> temperatures [2]
    #      ▼
    #   run_model：
    #      prefill / enforce_eager / bs>512  ──> self.model(input_ids, positions)，逐个发射kernel
    #      decode                            ──> 拷进CUDA Graph的静态缓冲区，graph.replay()
    #      │
    #      ├──> hidden_states [5, H]                     28层Transformer的输出，每个token一行
    #      │      │ compute_logits（lm_head）：prefill只取每条序列最后一个token；tp>1时gather到rank 0
    #      │      ▼
    #      └──> logits [2, V]                            每条序列在整个词表上的打分
    #      │
    #      ▼
    #   sampler(logits, temperatures)  ──> token_ids [2]（GPU张量）
    #      │ .tolist()：拷回CPU，并等待GPU全部算完
    #      ▼
    #   [15, 8]（Python列表）──> 返回给 LLMEngine.step ──> scheduler.postprocess 追加到各序列
    #
    #   最后 reset_context() 清空全局元数据，避免下一步误用
    #   其他rank执行同样的run()（同样的前向，参与all_reduce/gather），但不采样，返回None
    #
    # 执行一次step：准备输入 -> 前向 -> 采样，返回每条序列新生成的token id（长度为B的list，仅rank 0）
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 只有rank 0拿到完整的logits，采样只在rank 0上进行
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # logits: [B, V]
        logits = self.run_model(input_ids, positions, is_prefill)
        # sampler: [B, V] -> [B]；.tolist()会触发GPU->CPU同步，等待前向和采样完成
        # .tolist()：张量 -> Python列表，例如 tensor([15, 8]) -> [15, 8]
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # 为decode阶段捕获不同batch大小的CUDA Graph
    # CUDA Graph会把一次前向中所有kernel的启动参数（含张量地址）录制下来，之后replay只需一次启动
    # 要求：输入输出张量地址固定、形状固定，所以要预先分配静态缓冲区，并为每种batch大小单独捕获
    #
    #   静态缓冲区（只分配一次，max_bs行）        录制的graph
    #   input_ids    [max_bs]          ──┬──> graph[512]：读写前512行
    #   positions    [max_bs]            ├──> graph[496]：读写前496行
    #   slot_mapping [max_bs]            ├──> ...
    #   context_lens [max_bs]            ├──> graph[2]：读写前2行
    #   block_tables [max_bs, 最大block数] └──> graph[1]：读写前1行
    #   outputs      [max_bs, H]  <── 所有graph的输出都写到这里
    #
    # @torch.inference_mode()：装饰器写法，等价于在函数体外包一层 with torch.inference_mode():
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # 捕获的最大batch
        max_bs = min(self.config.max_num_seqs, 512)
        # 单条序列最多需要的block数，例如max_model_len=4096，BS=256 -> 16
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 静态输入输出缓冲区，所有graph共用（按[:bs]切片）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)                          # [max_bs]
        positions = torch.zeros(max_bs, dtype=torch.int64)                          # [max_bs]
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)                       # [max_bs]
        context_lens = torch.zeros(max_bs, dtype=torch.int32)                       # [max_bs]
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)       # [max_bs, max_num_blocks]
        outputs = torch.zeros(max_bs, hf_config.hidden_size)                        # [max_bs, H]
        # 捕获的batch大小：1、2、4、8，之后每隔16一档，直到max_bs
        # 例如max_bs=512：[1, 2, 4, 8, 16, 32, 48, ..., 512]，共36个graph
        # 小batch分得细，减少padding浪费；大batch间隔大，减少捕获时间和显存
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # batch大小 -> CUDAGraph
        self.graphs = {}
        # 所有graph共享的显存池
        self.graph_pool = None

        # 从大到小捕获：第一个（最大的）graph申请的内存池最大，后续小graph共享这个内存池即可，节省显存
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 先正常跑一遍：触发torch.compile、cuBLAS句柄等懒初始化，这些操作不能出现在捕获过程中
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # with语句块内的GPU操作不会真正执行，而是被录制进graph
            # self.graph_pool：录制时申请的显存从这个内存池分配；第一次为None，表示新建一个池
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存静态缓冲区，run_model中往里面填数据后replay
        # dict(a=1, b=2) 等价于 {"a": 1, "b": 2}
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
