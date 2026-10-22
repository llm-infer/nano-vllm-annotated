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


# 推理的入口，自己不做计算，而是把tokenizer、scheduler、modelrunner串起来
# 整体架构（以tp=2为例）：
#                  主进程（rank 0）                                       子进程（rank 1）
#   ┌──────────────────────────────────────────┐                  ┌─────────────────────────┐
#   │ LLMEngine                                │                  │ ModelRunner (rank=1)    │
#   │    ├─ tokenizer     文本 <-> token id     │   共享内存+Event  │   └─ loop() 死循环       │
#   │    ├─ scheduler     决定每步算哪些seq       │  ─────────────> │      等待指令 -> 执行      │
#   │    │    └─ block_manager   管理KV cache   │                 │                          │
#   │    ├─ model_runner  前向+采用，持有一半权重  │ <═══ NCCL ═════>│      持有一半的模型权重     │
#   └──────────────────────────────────────────┘    all_reduce   └──────────────────────────┘
# generate流程：
#   用户 -> generate() -> LLMEngine
#                           | add_requests：文本 -> tokens -> Sequence -> 放进调度器
#                           | step 循环：
#                           |   ① Scheduler.schedule()：选出要处理的序列，prefill还是decode
#                           |   ② ModelRunner.run()：执行计算、采样，得到每条序列的新token
#                           |   ③ Scheduler.postprocess()：把token追加到序列、判断是否结束、回收kv block
#                           └─ 所有序列都结束 -> 解码成文本返回
class LLMEngine:

    def __init__(self, model, **kwargs):
        # ---------- 1. 构造 Config ----------
        # 只挑属于Config字段的那部分参数，其他参数忽略
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 把kv block大小设置成Sequence的类变量，之后所有Sequence都按照这个大小计算自己占几个block
        Sequence.block_size = config.kvcache_block_size
        # ---------- 2. 拉起其余 TP rank 的子进程 ----------
        # 子进程对象列表，退出时要join，主进程确保子进程都退出后再进行后续工作
        self.ps = []
        # 每个子进程一个event，rank0调用event.set()通知子进程“共享内存里有新指令”
        self.events = []
        # 使用spawn的方式创建子进程
        #   fork：复制父进程的整个内存，包括CUDA上下文，复制后不可用
        #   spawn：启动一个全新的python解释器，环境干净，但启动更慢
        ctx = mp.get_context("spawn")
        # 为rank 1~N-1各启动一个ModelRunner子进程
        for i in range(1, config.tensor_parallel_size):
            # 每个子进程一个Event，rank 0写完共享内存后，通过event通知子进程有新指令
            event = ctx.Event()
            # 子进程直接以ModelRunner构造函数为入口，构造完成后进入loop()等待rank 0的指令
            # target：子进程要执行的函数（类也可以当做函数，即执行ModelRunner(config, i, event)）
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank0 启动 ModelRunner、tokenizer、scheduler
        # rank0 持有子进程所有events
        self.model_runner = ModelRunner(config, 0, self.events)
        # use_fast=True：使用rust实现的fast tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 必须在ModelRunner之后创建：config.num_kvcache_blocks在ModelRunner.allocate_kv_cache中才能确定
        # ModelRunner和Scheduler拿到的是同一个config对象，Scheduler能看到ModelRunner对config的修改
        self.scheduler = Scheduler(config)
        # 注册一个函数，python解释器正常退出时调用它
        atexit.register(self.exit)

    def exit(self):
        # 通知所有子进程执行exit
        self.model_runner.call("exit")
        # 删除这个引用，ModelRunner对象没有其他引用后会被回收，释放资源
        del self.model_runner
        # join()：主进程会阻塞等待子进程结束
        for p in self.ps:
            p.join()

    # 添加一个请求到scheduler，可以是字符串或者token id列表
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 字符串 -> token 
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 执行一次step，返回本步结束的序列和性能统计数据
    def step(self):
        # 获取本次step需要计算的seqs
        seqs, is_prefill = self.scheduler.schedule()
        # 通过正负号区分prefill和decode
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # 离线批量生成接口：一次性提交所有请求，循环step直到全部完成
    # prompts：文本列表或token id列表
    # sampling_params：单个（所有请求共用）或与prompts等长的列表
    # 返回：与prompt顺序一致的列表，每个元素为{"text": "生成的文本", "token_ids": "生成的token id列表"}
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 进度条，total为总数，每完成一条请求update(1)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # seq_id -> 生成的token id列表
        outputs = {}
        # 最近一次prefill/decode吞吐，展示在进度条里
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            # 高精度计时器，返回秒数
            t = perf_counter()
            output, num_tokens = self.step()
            # 吞吐 = 本step处理的token数 / 耗时
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # token id列表 -> 文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
