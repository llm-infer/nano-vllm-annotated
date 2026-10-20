import torch
from torch import nn


# 根据logits和temperature采样下一个token
# logits: [B, V]，temperatures: [B] -> sample_tokens: [B]（int64，每条序列一个token id）
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperatures.unsqueeze(1): [B] -> [B, 1]，广播到[B, V]，每行除以各自的温度
        #   例如 B=2，temperatures=[0.5, 1.0] -> [[0.5], [1.0]]
        #   logits第0行全部除以0.5，第1行全部除以1.0
        #   如果不unsqueeze，[B, V] / [B] 会让最后一维对齐（V对B），形状不匹配
        # .float()：转成float32；div_：原地除法（结尾下划线 = 原地修改，不分配新显存）
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 在词表维度做softmax得到概率分布：[B, V]
        #   softmax(x)_i = e^(x_i) / Σ_j e^(x_j)，每行的概率之和为1
        probs = torch.softmax(logits, dim=-1)
        # 用"指数分布竞争"（Gumbel-max的等价形式）代替torch.multinomial采样：
        #   对每个候选词i，采样 E_i ~ Exp(1)，取 argmax(p_i / E_i)，则选中i的概率恰好为p_i
        #   直观理解：E_i / p_i 服从速率为p_i的指数分布，看作"词i的到达时间"，p_i越大越早到达，最早到达者胜出
        # 好处：只用到逐元素运算和argmax，没有数据相关的控制流，容易被torch.compile融合，比multinomial快
        # torch.empty_like(probs).exponential_(1)：[B, V]，每个元素独立采样自Exp(1)
        # clamp_min_(1e-10)：防止E_i为0导致除零
        # probs / E：[B, V] -> argmax(dim=-1)：[B]
        # argmax(dim=-1)：每行最大值所在的下标，即被选中的token id
        #
        # 数值例子（V=3，一行）：
        #   probs = [0.6, 0.3, 0.1]
        #   E     = [1.5, 0.2, 0.4]（随机采样）
        #   probs/E = [0.4, 1.5, 0.25] -> argmax = 1，选中token 1
        #   概率小的token只要"运气好"（E很小）也可能被选中，多次采样的频率恰好等于probs
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
