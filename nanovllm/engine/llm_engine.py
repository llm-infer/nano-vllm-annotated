import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


# 推理引擎入口，串联 tokenizer、Scheduler（调度）、ModelRunner（模型执行）
#
# 整体架构（以 tensor_parallel_size=2 为例）：
#
#   主进程（rank 0）                                                 子进程（rank 1）
#   ┌─────────────────────────────────────────────┐                  ┌──────────────────────────┐
#   │ LLMEngine                                   │                  │ ModelRunner (rank=1)     │
#   │   ├─ tokenizer     文本 <-> token id        │                  │   └─ loop() 死循环：     │
#   │   ├─ scheduler     决定每步算哪些序列       │  共享内存+Event  │        等待指令 -> 执行  │
#   │   │    └─ block_manager  管理KV cache块     │  ─────────────>  │                          │
#   │   └─ model_runner (rank=0)                  │                  │   持有一半的模型权重     │
#   │          前向 + 采样，持有一半的模型权重    │ <═══ NCCL ═════> │                          │
#   └─────────────────────────────────────────────┘   (all_reduce)   └──────────────────────────┘
#
# generate() 的主循环：
#
#   add_request × N（所有请求进入 waiting 队列）
#          │
#          ▼
#   ┌──> is_finished()? ──是──> 按 seq_id 排序，detokenize，返回结果
#   │      │否
#   │      ▼
#   │   step():
#   │     1. scheduler.schedule()       选出本步的序列，决定是 prefill 还是 decode
#   │     2. model_runner.call("run")   构造输入 -> 模型前向 -> 采样，得到每条序列的新 token
#   │     3. scheduler.postprocess()    把新 token 追加到序列，判断是否结束，释放结束序列的 KV
#   │      │
#   └──────┘
class LLMEngine:

    # model：模型目录路径
    # **kwargs：可变关键字参数，调用时写的 key=value 都会被收集到一个字典 kwargs 里
    #   例如 LLMEngine(path, enforce_eager=True, max_model_len=4096)
    #   -> kwargs = {"enforce_eager": True, "max_model_len": 4096}
    def __init__(self, model, **kwargs):
        # 从kwargs中挑出Config里定义过的字段，其余参数忽略
        # 这样LLM(path, enforce_eager=True, ...)可以直接透传给Config
        # fields(Config)：dataclasses提供的函数，返回dataclass所有字段的描述对象，field.name 是字段名
        # {... for ...}：集合推导式，得到 {"model", "max_num_batched_tokens", ...}
        config_fields = {field.name for field in fields(Config)}
        # {k: v for ...}：字典推导式，只保留 Config 认识的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # **config_kwargs：把字典"解包"成关键字参数，等价于 Config(model, enforce_eager=True, ...)
        config = Config(model, **config_kwargs)
        # 同步block大小到Sequence的类变量，用于计算num_blocks等
        # 修改类变量（Sequence.xxx）会影响所有实例，之后创建的每个Sequence都使用这个值
        Sequence.block_size = config.kvcache_block_size
        # 子进程列表（rank 1~N-1），退出时join
        self.ps = []
        # 与子进程一一对应的Event，rank 0用它通知子进程"共享内存里有新指令"
        # Event 是跨进程的"信号灯"：set() 亮灯，wait() 阻塞直到灯亮，clear() 熄灯
        self.events = []
        # 张量并行：rank 0运行在当前主进程，rank 1~N-1各起一个子进程
        # 必须用spawn而不是fork：CUDA在fork出来的子进程中无法重新初始化
        #   fork：复制父进程的整个内存（包括已初始化的CUDA状态，复制后不可用）
        #   spawn：启动一个全新的Python解释器，重新import模块，干净但启动更慢
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            # 每个子进程一个Event，rank 0写完共享内存后set，通知子进程读取指令
            event = ctx.Event()
            # 子进程直接以ModelRunner构造函数为入口，构造完成后进入loop()等待rank 0的指令
            # target：子进程要执行的函数（类也可以当函数调用，即执行 ModelRunner(config, i, event)）
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0持有所有子进程的Event列表
        self.model_runner = ModelRunner(config, 0, self.events)
        # 分词器：文本 <-> token id；use_fast=True使用Rust实现的fast tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 必须在ModelRunner之后创建：config.num_kvcache_blocks在ModelRunner.allocate_kv_cache中才确定
        # （ModelRunner和这里拿到的是同一个config对象，ModelRunner对它的修改这里能看到）
        self.scheduler = Scheduler(config)
        # 进程退出时自动清理，通知子进程退出、释放共享内存和进程组
        # atexit.register(f)：注册一个函数，Python解释器正常退出时自动调用它
        atexit.register(self.exit)

    def exit(self):
        # call("exit")会同时通知所有子进程执行exit
        self.model_runner.call("exit")
        # del：删除这个引用，ModelRunner对象没有其他引用后会被回收，释放显存
        del self.model_runner
        # join()：阻塞等待子进程结束
        for p in self.ps:
            p.join()

    # 添加一个请求：字符串先tokenize，也可以直接传token id列表
    # str | list[int]：类型注解，表示"字符串或整数列表"，只是提示，Python运行时不检查
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # isinstance(x, T)：判断x是否是T类型
        if isinstance(prompt, str):
            # "你好" -> [108386]，文本切分成token并映射为id
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 执行一次推理step
    # 返回：(本步结束的序列 [(seq_id, 生成的token列表), ...], 吞吐统计用的token数)
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # 用于统计吞吐：prefill返回本次计算的token数（正数），decode返回序列数的相反数（负数）
        # 通过正负号区分prefill和decode，调用方无需额外返回is_prefill
        # A if cond else B：三元表达式，cond为True取A，否则取B
        # sum(x for x in ...)：生成器表达式求和，不会先构造完整列表
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 收集本step结束的序列
        # [表达式 for x in 列表 if 条件]：列表推导式，筛选+变换一步完成
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # 离线批量生成接口：一次性提交全部请求，循环step直到全部完成
    # prompts：文本列表或token id列表
    # sampling_params：单个（所有请求共用）或与prompts等长的列表
    # 返回：与prompts顺序一致的列表，每个元素为 {"text": 生成文本, "token_ids": 生成的token id列表}
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # tqdm：进度条库，total为总数，每完成一条请求update(1)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 传入单个SamplingParams时，所有请求共用
        # [x] * n：把列表重复n次，得到n个指向同一对象的引用
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # zip(a, b)：把两个列表按位置配对，(a[0], b[0]), (a[1], b[1]), ...
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # seq_id -> 生成的token id列表
        outputs = {}
        # 最近一次prefill/decode step的吞吐（tok/s），用于进度条显示
        # 0. 即浮点数0.0
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            # perf_counter()：高精度计时器，返回秒数，两次相减得到耗时
            t = perf_counter()
            output, num_tokens = self.step()
            # 吞吐 = 本step处理的token数 / 本step耗时
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # decode每条序列生成1个token，token数即序列数
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 序列完成的顺序与提交顺序不同，按seq_id（即提交顺序）排序还原
        # 例如：outputs = {2: [...], 0: [...], 1: [...]} -> 按0, 1, 2的顺序取出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # decode：token id列表 -> 文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
