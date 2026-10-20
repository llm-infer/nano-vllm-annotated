from nanovllm.engine.llm_engine import LLMEngine


# 对外暴露的接口类，与vLLM的用法保持一致：from nanovllm import LLM
# 所有功能都在LLMEngine中实现
class LLM(LLMEngine):
    pass
