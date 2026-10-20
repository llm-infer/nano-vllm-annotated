import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


# 最简单的使用示例：对两个prompt进行生成
def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # enforce_eager=True：不捕获CUDA Graph，启动更快，便于调试
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 套用对话模板，把用户输入包装成模型训练时的对话格式
    # tokenize=False：返回字符串而不是token id，交给引擎内部再tokenize
    # add_generation_prompt=True：在末尾加上assistant回复的起始标记，让模型从这里开始回答
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
