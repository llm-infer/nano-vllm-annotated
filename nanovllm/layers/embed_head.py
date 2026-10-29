import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# 实际用法
#   VocabParallelEmbedding -> Qwen3Model.embed_tokens: token id [N] -> 向量 [N, 1024]
#   ParallelLMHead         -> Qwen3ForCasualLM.lm_head: hidden [N, 1024] -> logits [B, 151936] -> token id [N]
#   两者形状都是 [151936, 1024], Qwen3-0.6B 里共用一份权重(tie_word_embeddings)
#   embedding 是“按token id取一行”，lm_head是和每一行做点积，用的是同一张表
#
# 词表并行的Embedding：把词表按行均分到各rank，每个tank只存 V/tp 行
# 例如：vocab=151936，tp=2 -> rank=0存[0, 75968], rank1存[75968, 151936]
#   完整权重 [V, H]
#   本地权重 [V/tp, H]
#   x [N] (token id) -> y [N, H]
#
# Embedding是什么：一张 [V, H] 表，第i行是token i的向量表示，查表就是按照token id取对应的行
#   例如 V=4，H=2，weight = [[1, 2]
#                           [3, 4]
#                           [5, 6]
#                           [7, 8]]
#   x = [3, 0] -> y = [[7, 8]  取token id=3和token id=0这两个token的向量
#                      [1, 2]]
#
# tp=2时的查表过程（rank0负责token 0~1，rank1负责2~3）
#   输入 x=[1, 3]
#                rank 0（负责0~1）                     rank 1（负责2~3）
#   mask:       [True, False]                          [False, True]
#   本地下标      [1, 0]（2不归我管，映射成0）              [0, 1]（1不归我管，映射成0；token3在rank1本地下标是3-2=1）
#   查表         [e1, 本地第0行]                          [本地第0行，e3]
#   乘mask清零    [e1, 0]                                [0, e4]
#                   └────────── all_reduce 求和 ──────────┘
#   结果                           [e1, e4]
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 当前rank编号和张量并行总数
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 词表需要均分到各rank
        assert num_embeddings % self.tp_size == 0
        # 完整的词表大小
        self.num_embeddings = num_embeddings
        # 每个rank负责的词数量 V/tp
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前rank负责的词表区间 [vocab_start_idx, vocab_end_idx]
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 权重占位 [V/tp, H]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 给权重挂上加载函数，load_model时会调用它来切分权重
        self.weight.weight_loader = self.weight_loader

    # 从完整权重中按行切出当前rank负责的部分 [V, H] -> [V/tp, H]
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # Tensor.narrow：沿某一维截取一段连续切片，返回一个 view
        # 以tp=2，完整权重形状[V, H]为例，shard_size = V/2
        #
        #                    dim 1 (H) -->          
        #           ┌──────────────────────────────┐  ──┐
        #           │                              │    │
        #           │          rank 0              │   shard_size
        # dim 0 (V) │                              │    │
        #     |     │──────────────────────────────│  ──┘
        #     v     │                              │
        #           │          rank 1              │
        #           │                              │
        #           └──────────────────────────────┘
        # 注意：narrow返回的是view，只修改offset/size元数据，不拷贝数据；真正拷贝数据发生在copy_
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 数据CPU -> GPU：loaded_weight在内存，param在显存
        param_data.copy_(loaded_weight)

    # x [N] -> y [N, H]
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # mask [N] bool, 表示该token是否落在本rank的词表区间
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # x - self.vocab_start_idx：x 的元素是全局 token id，而rank本地只存了部分权重，而且下标是从0开始的，所以要先转为本地 token id
            # bool张量参与乘法运算时，True=1，False=0，所以不属于本rank词表区间的token会被置为0
            x = mask * (x - self.vocab_start_idx)
        # 查表：x [N], weight [V, H] -> y [N, H]
        # 等价于weight[x]：用x中的每个整数作为行号取出对应的行
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # mask.unsqueeze(1)：往mask [N] 第1维加一个维度 -> [N, 1]，用于广播
            # [N, 1] * [N, H] -> [N, H] 在第1维上广播
            y = mask.unsqueeze(1) * y
            # 逐位相加求和，[N, H] + [N, H]
            # 每个token只在一个rank上有非零结果，求和后每个rank得到完整的embedding
            #
            #  [[0,0,0]             [[1,1,1]            [[1,1,1]
            #   [0,0,0]      +       [2,2,2]     =       [2,2,2]
            #   [4,5,6]              [0,0,0]             [4,5,6]
            #   [7,8,9]]             [0,0,0]]            [7,8,9]]
            #
            #   rank0 y              rank1 y          dist.all_reduce(y)
            dist.all_reduce(y)
        return y


# 输出层：hidden_states [N, H] -> 词表上的logits [N, V]
# 同样按词表切分，每个rank算出一部分词的logits，再汇总到rank 0
#   本地权重：[V/tp, H]
#   x：[N, H] -> (prefill取每条序列最后的token) [B, H] -> 计算部分词的logits [B, V/tp] -> gather [B, V]
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
        # prefill 时，x包含了所有序列本次前向计算的所有token
        #       x = [103 43 542 4 29 40 53 98]
        #     pos = [2    3   4 0  1  2  3  4]
        #                     ↑             ↑
        #                    需要           需要
        # lm_head只需要序列最后一个token预测下一个token，剔除掉不用的token可大幅减少计算量
        # cu_seqlen_q = [0, 3, 8]：每条序列在拼接张量x的边界
        # cu_seqlen_q[1:] - 1 = last_indices = [2, 7]：每条序列最后一个token在拼接张量x的下标
        # x = x[last_indices]：取出最后一个token [N, H] -> [B, H]
        # 
        # decode阶段，每个序列只有一个token，x本来就是 [B, H]
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            # .contiguous()：确保内存连续，后续矩阵乘法更高效
            x = x[last_indices].contiguous()
        # [B, H] @ [V/tp, H]^T -> [B, V/tp]
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 只有rank0需要做采样，所以用gather，而不是all_gather，减少通信
            # all_logits：rank 0上是tp个[B, V/tp]，其他rank 为None
            # torch.empty_like：创建相同形状、dtype、设备的未初始化张量
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # logits：本rank要发出去的数据
            # all_logits：接收缓冲区列表，只有 dst 需要
            # 0：目标rank
            # 
            # dist.gather(logits, all_logits, 0)流程
            #   rank 0                          rank 1
            #   all_logits = [buf0, buf1]       all_logits = None
            #   
            #    +--------+                      +--------+
            #    |   L0   |                      |   L1   |
            #    +--------+                      +--------+
            #        |                               |
            #        | local copy                    | send
            #        v                               |
            #   buf0 = L0                            |
            #   buf1 = L1 <--------------------------+
            #   dist.gather会保证rank x的数据写到 buf[x]里
            #
            # rank 0: torch.cat([buf0, buf1], dim=-1)
            #    [B, V/2], [B, V/2] -> [B,V]
            #            0                75968            151936
            #            +----------------+----------------+
            #    seq0    |       L0       |       L1       |
            #    seq1    |                |                |
            #            +----------------+----------------+
            #            
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
