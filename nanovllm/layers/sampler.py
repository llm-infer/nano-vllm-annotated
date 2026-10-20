import torch
from torch import nn


# 根据logits和temperature采样下一个token
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: [batch, vocab]，temperatures: [batch]，unsqueeze成[batch, 1]按行缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 用"指数分布竞争"（Gumbel-max的等价形式）代替torch.multinomial采样：
        #   对每个候选词i，采样 E_i ~ Exp(1)，取 argmax(p_i / E_i)，则选中i的概率恰好为p_i
        # 好处：只用到逐元素运算和argmax，没有数据相关的控制流，容易被torch.compile融合，比multinomial快
        # clamp_min_(1e-10)：防止E_i为0导致除零
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
