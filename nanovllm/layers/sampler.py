import torch
from torch import nn


# 根据logits和template采样下一个token
# logits：[B, V], temperatures：[B] -> sample_tokens [B]（每个序列一个token id）
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperatures.unsqueeze(1) -> [B] -> [B, 1] 广播到 [B, V]
        #   例如 B=2，temperatures=[0.5, 1.0] -> [[0.5], [1.0]]
        #   logits 第 0行全部除以0.5, 第1行全部除以1.0
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 在词表维度做softmax得到概率分布 [B, V]
        #   softmax(x)_i = e^(x_i) / Σ_j e^(x_j)，每行的概率之和为1
        probs = torch.softmax(logits, dim=-1)
        # torch.empty_like(probs).exponential_(1)：[B, V]，每一行按"指数分布"来随机，每行总和为1
        # clamp_min_(1e-10)：设一个下限，小于1e-10的数，一律抬到1e-10
        # argmax(dim=-1)：每行最大值所在的下标，即被选中的token id
        # 好处
        #   避免固定的输出：加入随机因素，让概率小的token有机会被选中，而不是每次取概率最大的值
        #   整体概率不变：多次采用的频率依旧等于probs
        #   易于并行：只用到element-wise和argmax，没有数据相关的控制流，易于被torch.compile融合和并行计算
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
