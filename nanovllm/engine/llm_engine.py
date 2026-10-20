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
# 一次step的流程：schedule选出序列 -> model_runner前向+采样 -> postprocess更新序列状态
class LLMEngine:

    def __init__(self, model, **kwargs):
        # 从kwargs中挑出Config里定义过的字段，其余参数忽略
        # 这样LLM(path, enforce_eager=True, ...)可以直接透传给Config
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 同步block大小到Sequence的类变量，用于计算num_blocks等
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        # 张量并行：rank 0运行在当前主进程，rank 1~N-1各起一个子进程
        # 必须用spawn而不是fork：CUDA在fork出来的子进程中无法重新初始化
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            # 每个子进程一个Event，rank 0写完共享内存后set，通知子进程读取指令
            event = ctx.Event()
            # 子进程直接以ModelRunner构造函数为入口，构造完成后进入loop()等待rank 0的指令
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0持有所有子进程的Event列表
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 必须在ModelRunner之后创建：config.num_kvcache_blocks在ModelRunner.allocate_kv_cache中才确定
        self.scheduler = Scheduler(config)
        # 进程退出时自动清理，通知子进程退出、释放共享内存和进程组
        atexit.register(self.exit)

    def exit(self):
        # call("exit")会同时通知所有子进程执行exit
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # 添加一个请求：字符串先tokenize，也可以直接传token id列表
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 执行一次推理step
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # 用于统计吞吐：prefill返回本次计算的token数（正数），decode返回序列数的相反数（负数）
        # 通过正负号区分prefill和decode，调用方无需额外返回is_prefill
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 收集本step结束的序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # 离线批量生成接口：一次性提交全部请求，循环step直到全部完成
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 传入单个SamplingParams时，所有请求共用
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
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
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
