import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


# 整除，并断言能整除（张量并行要求维度能被tp_size整除）
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# 线性层基类，支持张量并行（Megatron-LM风格）
# 权重形状为[output_size, input_size]，与nn.Linear一致，计算 y = x @ W^T + b
# tp_dim：权重沿哪个维度切分，0为按输出维切（列并行），1为按输入维切（行并行），None为不切
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 这里的input_size/output_size已经是切分后的本地大小
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 挂上子类实现的weight_loader，load_model时调用它从完整权重中切出本rank的部分
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 注册为None，保证self.bias属性存在
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 不切分的线性层，每个rank持有完整权重（本模型中未使用）
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# 列并行：按输出维切分权重，每个rank计算输出的一部分
# 输入x是完整的，输出是切分的，不需要通信
# 例如：W[4096, 1024]，tp=2 -> 每个rank持有W[2048, 1024]，输出各自的2048维
# 通常与后面的行并行配对使用：列并行的切分输出直接作为行并行的切分输入，中间无需通信
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    # 沿第0维（输出维）切出第tp_rank份
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# 合并的列并行层：把多个输入相同的线性层（如MLP的gate_proj和up_proj）合并成一个，一次矩阵乘法完成
# 每个rank的本地权重布局为 [gate的第rank份; up的第rank份]
# 这样本地输出前一半是gate、后一半是up，SiluAndMul直接chunk即可
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        # 各个子层的输出大小，如[intermediate_size, intermediate_size]
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # loaded_shard_id：当前加载的是第几个子层（0为gate，1为up）
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 该子层在本地权重中的起始偏移和大小（都是切分后的）
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 把完整子层权重沿输出维切成tp_size份，取本rank那份
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# QKV合并的列并行层：把q_proj、k_proj、v_proj合并成一个
# 按注意力头切分：每个rank负责 num_heads/tp 个q头 和 num_kv_heads/tp 个kv头
# 本地权重布局：[本rank的q头; 本rank的k头; 本rank的v头]
# GQA时kv头数少于q头数，所以q和k/v的分片大小不同
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 未指定kv头数时退化为普通多头注意力（MHA）
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 本rank的头数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 合并后的总输出维度（切分前）
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    # loaded_shard_id："q"、"k"、"v"，决定写入本地权重的哪一段
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# 行并行：按输入维切分权重，每个rank用切分后的输入计算出完整形状的部分和，再all_reduce求和
# 例如：W[1024, 4096]，tp=2 -> 每个rank持有W[1024, 2048]，输入是列并行输出的那2048维
# 用于注意力的o_proj和MLP的down_proj
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # bias是一维的，按输出维不切分，每个rank都保存完整的bias
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        # 沿第1维（输入维）切出第tp_rank份
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias只在rank 0加，否则all_reduce后bias会被加tp_size次
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
