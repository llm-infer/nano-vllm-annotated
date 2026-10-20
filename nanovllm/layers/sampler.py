import torch
from torch import nn


# 根据logits和temperature采样下一个token
# logits: [B, V]，temperatures: [B] -> sample_tokens: [B]（int64，每条序列一个token id）
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperatures.unsqueeze(1): [B] -> [B, 1]，广播到[B, V]，每行除以各自的温度
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 在词表维度做softmax得到概率分布：[B, V]
        probs = torch.softmax(logits, dim=-1)
        # 用"指数分布竞争"（Gumbel-max的等价形式）代替torch.multinomial采样：
        #   对每个候选词i，采样 E_i ~ Exp(1)，取 argmax(p_i / E_i)，则选中i的概率恰好为p_i
        #   直观理解：E_i / p_i 服从速率为p_i的指数分布，看作"词i的到达时间"，p_i越大越早到达，最早到达者胜出
        # 好处：只用到逐元素运算和argmax，没有数据相关的控制流，容易被torch.compile融合，比multinomial快
        # torch.empty_like(probs).exponential_(1)：[B, V]，每个元素独立采样自Exp(1)
        # clamp_min_(1e-10)：防止E_i为0导致除零
        # probs / E：[B, V] -> argmax(dim=-1)：[B]
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
