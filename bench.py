import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


# 吞吐测试：256条随机长度的请求，统计总生成token数 / 总耗时
def main():
    # 固定随机种子，保证每次测试的输入一致，结果可比
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 开启CUDA Graph，测试真实性能
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 随机token作为输入，长度在[100, 1024]之间
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # ignore_eos=True：遇到EOS也不停止，保证每条请求都生成恰好max_tokens个token，总生成量固定
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 预热一次，排除首次运行的编译、初始化等开销
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    # 只统计生成的token数（不含prompt）
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
