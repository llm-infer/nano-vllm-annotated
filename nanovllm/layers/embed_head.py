import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# 词表并行的Embedding：把词表按行均分到各个rank，每个rank只存 V/tp 行
# 例如：vocab=151936，tp=2 -> rank0存[0, 75968)，rank1存[75968, 151936)
#   完整权重：[V, H]
#   本地权重：[V/tp, H]
#   x: [N] (token id) -> y: [N, H]
#
# Embedding是什么：一张 [V, H] 的表，第i行是token i的向量表示，"查表"就是按token id取对应的行
#   例如 V=5，H=2，weight = [[.1,.2], [.3,.4], [.5,.6], [.7,.8], [.9,1.]]
#   x = [3, 0] -> y = [[.7,.8], [.1,.2]]
#
# tp=2时的查表过程（V=6，rank0负责token 0~2，rank1负责token 3~5，输入x=[1, 4]）：
#
#                rank 0（负责0~2）                     rank 1（负责3~5）
#   mask:        [True,  False]                        [False, True ]
#   本地下标:    [1, 0]（4不归我，映射成0）            [0, 1]（1不归我，映射成0；4-3=1）
#   查表:        [e1, 本地第0行]                       [本地第0行, e4]
#   乘mask清零:  [e1, 0]                               [0, e4]
#                   └────────── all_reduce 求和 ──────────┘
#   结果（两个rank相同）：[e1, e4]
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 当前rank编号与张量并行总数
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        # 完整词表大小V
        self.num_embeddings = num_embeddings
        # 每个rank负责的词数 V/tp
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前rank负责的词表区间[vocab_start_idx, vocab_end_idx)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # weight: [V/tp, H]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 给参数挂上自定义加载函数，load_model时会调用它来切分权重
        self.weight.weight_loader = self.weight_loader

    # 从完整权重中按行切出当前rank负责的部分：[V, H] -> [V/tp, H]
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # narrow返回视图，不拷贝
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # x: [N] -> y: [N, H]
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # mask: [N]，bool，表示该token是否落在本rank的词表区间
            # 不属于本rank的token id先映射为0（避免越界），属于的转成本地下标
            # 例如：rank1区间[75968, 151936)，x=[10, 80000] -> mask=[F, T]，x=[0, 4032]
            # 张量的比较运算逐元素进行，得到bool张量；& 是逐元素的"与"
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # bool张量参与乘法时 True=1、False=0，所以不属于本rank的位置被乘成0
            x = mask * (x - self.vocab_start_idx)
        # 查表：[N] -> [N, H]
        # F.embedding(x, weight) 等价于 weight[x]：用x中的每个整数作为行号取出对应的行
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # mask.unsqueeze(1): [N, 1]，广播到 [N, H]，把不属于本rank的token的结果清零
            #   广播（broadcast）：形状不同的张量做逐元素运算时，长度为1的维度会自动"复制"到对方的长度
            #   [N, 1] * [N, H]：每一行的那1个值与该行的H个值分别相乘
            #   如果不unsqueeze，[N] * [N, H] 会按最后一维对齐，变成N与H对齐，形状不匹配报错
            # 再all_reduce求和：每个token只在一个rank上有非零结果，求和后每个rank都得到完整的embedding
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


# 输出层（lm_head）：hidden_states -> 词表上的logits
# 同样按词表切分，每个rank算出一部分词的logits，再汇总到rank 0
#   本地权重：[V/tp, H]
#   x: [N, H] -> (prefill取最后token) [B, H] -> 本地logits [B, V/tp] -> gather -> [B, V]（仅rank 0）
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        # prefill时只需要每条序列最后一个token的logits来预测下一个token，
        # 其他位置的logits用不到，提前取出来可以大幅减少lm_head的计算量（词表通常十几万）
        # cu_seqlens_q[1:] - 1 即每条序列最后一个token在拼接张量中的下标
        # 例如：cu_seqlens_q=[0, 3, 103] -> last_indices=[2, 102]
        # x: [N, H] -> [B, H]
        # decode时每条序列只有一个token，x本来就是[B, H]
        #
        #   拼接的hidden_states（3条序列，长度3、5、2）：
        #   下标:  0  1  2 | 3  4  5  6  7 | 8  9
        #                ↑              ↑     ↑
        #   cu_seqlens_q = [0, 3, 8, 10] -> [1:] = [3, 8, 10] -> -1 = [2, 7, 9]
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            # x[整数张量]：高级索引，按下标一次取出多行，[N, H] -> [len(last_indices), H] = [B, H]
            # .contiguous()：确保结果在内存中连续排布（后续矩阵乘法更高效）；已经连续时不拷贝
            x = x[last_indices].contiguous()
        # [B, H] @ [V/tp, H]^T -> [B, V/tp]
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 只有rank 0需要做采样，所以用gather（汇总到rank 0）而不是all_gather，减少通信
            # all_logits：rank 0上是tp个 [B, V/tp] 的接收缓冲区，其他rank为None
            # torch.empty_like(t)：创建与t形状、dtype、设备相同的未初始化张量
            # for _ in range(n)：_ 表示不使用这个循环变量
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # gather：所有rank把自己的logits发给rank 0，rank 0按rank顺序放进all_logits
            #   rank0: [B, V/2] ─┐
            #                    ├─> rank 0 的 all_logits = [rank0的, rank1的]
            #   rank1: [B, V/2] ─┘
            dist.gather(logits, all_logits, 0)
            # 按词表维度拼接：tp个[B, V/tp] -> [B, V]
            # torch.cat(列表, dim)：沿dim维把多个张量首尾相接，其他维度必须相同；-1表示最后一维
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
