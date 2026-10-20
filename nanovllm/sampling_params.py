from dataclasses import dataclass


@dataclass(slots=True)
# 配置如何对模型输出的logits（每个词的分数）进行采样
class SamplingParams:
    # 对分数进行缩放，再计算softmax得到概率：softmax(logits / temperature)
    # temperature < 1，分布变尖，高概率的词更占优，输出更确定更保守
    # temperature > 1，分布变平，低概率词更容易被选中，输出更随机、更发散
    # temperature 趋向 0，每次都选概率最大的值，即贪心解码
    # 例如：
    #   候选词	 很    真	 不	  比
    #   logits	3.0	  2.0  1.0	0.0
    #   e的指数: 20.1, 7.4, 2.7, 1.0   (总和 31.2)
    #    概率:   64%,  24%,  9%,  3%
    # logits/0.5  6    4    2    0
    #   e的指数: 403, 54.6, 7.4, 1.0   (总和 466)
    #    概率:  86.5%,11.7%,1.6%,0.2%
    temperature: float = 1.0
    # 本次请求最多生成多少token（不含prompt）
    # 防止出现一直停不下来的情况
    max_tokens: int = 64
    # 是否忽略结束符
    # False：模型生成了 EOS token 就立刻停止, 这是正常的对话/补全行为
    # True：模型生成了 EOS token 仍继续，常用于性能测试时固定输入输出长度，便于做数据对比
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
