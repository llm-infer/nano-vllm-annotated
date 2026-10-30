import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    # nn.Module：Pytorch中的模型基类
    #   继承它以后，__init__中赋值的nn.Parameter和子module会自动注册
    #   model.parameters、model.modules、model.get_parameter都依赖注册信息
    #   调用layer(x)会自动执行layer.forward(x)
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # 权重沿哪个维度切分：0切output_size，即切输出维度，1切input_size，即切输入维度
        self.tp_dim = tp_dim
        # 当前rank编号，决定用完整权重中的第几份
        self.tp_rank = dist.get_rank()
        # 张量并行总数
        self.tp_size = dist.get_world_size()
        # 这里的output_size/input_size是切分后的本地大小
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 挂上子类实现的weight_loader，负责从完整权重中切出本rank的部分
        self.weight.weight_loader = self.weight_loader
        if bias:
            # bias [output_size]
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    # 基类不实现forward，子类必须重写
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 不切分的线性层，每个rank持有完整的权重
# weight：[output_size, input_size]
# x: [N, input_size] -> y: [N, output_size]
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
        # F.linear(x, W, b) = x @ W^T + b
        return F.linear(x, self.weight, self.bias)


# 列并行：按输出维度切分，每个rank计算输出的一部分
# 输入x是完整的，输出是切分的
#   完整权重: [output_size， input_size]
#   本地权重：[output_size/tp, input_size]
#   x: [N, input_size] -> y: [N, output_size/tp]
# 用法：通常与行并行（输入x是切分）配合使用，全程不需要通信
#   完整x -> 列并行 -> 分片y -> 行并行（输入是切分的，列并行的切分输出天然可以作为行并行的切分输入）
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        # 延输出维度，切出第rank份权重
        start_idx = self.tp_rank * shard_size
        # [output_size, input_size] -> [output_size/tp, input_size]
        #       [[1,2,3]    ───┐rank0
        #        [4,5,6]    ───┘
        #        [7,8,9]    ───┐rank1
        #        [10,11,12]]───┘
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, input_size] -> [output_size/tp, input_size] -> [N, output_size/tp]
        return F.linear(x, self.weight, self.bias)


# 把多个"输入相同、各自列并行"的线性层合并成一个大矩阵，一次 GEMM 算完
# 以 gate_proj + up_proj = gate_up_proj为例, output_sizes = [I, I]
# 每个子矩阵各自按 tp 切分，再在本地按顺序拼接
#   gate_proj.weight      up_proj.weight
#     [g0]  ─┐rank0         [u0]  ─┐rank0
#     [g1]  ─┘              [u1]  ─┘
#     [g2]  ─┐rank1         [u2]  ─┐rank1
#     [g3]  ─┘              [u3]  ─┘
#
#   rank0本地参数 [2*I/tp, input_size]  rank1本地参数 [2*I/tp, input_size]
#     [g0] ┐ shard_id=0, offset=0      [g2] ┐ shard_id=0, offset=0
#     [g1] ┘                           [g3] ┘
#     [u0] ┐ shard_id=1, offset=2      [u2] ┐ shard_id=1, offset=2
#     [u1] ┘                           [u3] ┘
#
# forward后每个rank输出[N, gate_i | up_i], SiluAndMul 本地chunk即可
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        # 各子矩阵的完整输出维度，如 [I, I]，每项都需能被 tp_size 整除
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # 把 checkpoint中的某一个子矩阵切出当前rank的分片，写入本地合并权重属于它的那一段
    # 每个子矩阵调用一次，gate调用时loaded_shard_id=0，up调用时为1
    # param：本地合并权重，例如[2I/tp, input_size]
    # loaded_weight: checkpoint中完整的子矩阵，[output_sizes[id], input_size]，例如[I, input_size]
    # loaded_shard_id：子矩阵编号，决定写入本地param的哪一段
    # 示例：output_sizes=[4, 4]（I=4），tp=2，当前 rank=1
    #   本地 param 布局（4 行）：
    #           行 0~1: gate 的第 rank 份   <- shard_id=0 写这里
    #           行 2~3: up   的第 rank 份   <- shard_id=1 写这里
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # ---- 第 1 步：计算该子矩阵在本地 param 中的写入区间 ----
        # 本地 param 是各子矩阵分片按顺序拼接的：[sub0分片 | sub1分片 | ...]
        # 所以 offset = 前面所有子矩阵的分片大小之和
        #   shard_id=0: offset = 0                 = 0
        #   shard_id=1: offset = output_sizes[0]/tp = 4/2 = 2
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # shard_id=1，取2~3行
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # ---- 第 2 步：从完整子矩阵中切出当前 rank 的分片 ----
        # 沿输出维度切成tp份，取第rank份
        #   up_w [4, H]:  [u0]  ─┐ rank0
        #                 [u1]  ─┘
        #                 [u2]  ─┐ rank1  <- 取这份
        #                 [u3]  ─┘
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        # ---- 第 3 步：写入 ----
        # 两者shape都是[I/tp, input_size]，直接原地拷贝
        param_data.copy_(loaded_weight)


#  QKV合并的列并行层：把q_proj、k_proj、v_Proj合并成一个
# 按注意力头切分：每个rank负责Hq = total_heads/tp个q头和Hkv = total_kv_heads/tp个kv头
#   本地权重：[(Hq + 2Hkv) * head_dims, input_size]，布局为[q头, k头, v头]
#   x: [N, input_size] -> y [N, (Hq+Hk+Hv) * head_dims]
#
# 例如qwen3-0.6B：16个q头，8个kv头，head_dims=128，tp=2时
#        HF checkpoint中的权重       切成2份          rank0 qkv_proj.weight     rank1 qkv_proj.weight
#   q_proj [16*128, input_size] -> q头0~7|q头8~15       第一段：q头0~7              第一段：q头8~15  
#   k_proj [8*128, input_size]  -> k头0~3|k头4~7        第二段：k头0~3              第二段：q头8~15  
#   v_proj [8*128, input_size]  -> v头0~3|v头4~7        第三段：v头0~3              第三段：v头4~7
#                                                 [(8+4+4)*128, input_size]  [(8+4+4)*128, input_size]
#
# 每个rank拿到自己那几个头的qkv，注意力可在本rank上计算，不需要通信
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
        # 未指定kv_heads，则退化成MHA
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        # 每个头的维度
        self.head_size = head_size
        # 每个rank负责的q头数、kv头数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # qkv合并后的输出维度（切分前）
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    # loaded_shard_id：q、k、v，决定写入本地权重的哪些行
    # loaded_weight：q为[num_heads*head_dim, input_size], k/v为[num_kv_heads*head_dim, input_size]
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 本地权重行区间
        # q: [0, Hq*D]
        # k: [Hq*D, Hq*D+Hkv*D]
        # v: [Hq*D+Hkv*D, Hq*D+Hkv*D*2]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        # 切出q/k/v所在本地权重的那部分
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 切出qkv所在checkpoint的那部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# 行并行：按照输入维度切分，每个rank用切分后的输入计算完整形状的结果，再all_reduce求和
# 例如，一个完整的矩阵乘计算过程为  1*2 + 3*4 + 5*6 + 7*8 = 100
#                              1*2 + 3*4   5*6 + 7*8
#       将输入维度切分         rank0计算，14   rank1计算，86
#       求和                          14+86=100
# 完整权重 [output_size, input_size]
# 本地权重 [output_size, input_size/tp]
# x: [N, input_size/tp] -> 部分和 [N, output_size] -> all_reduce -> 最终结果 [N, output_size]
# 
# 适用场景：列并行"切输出维"，行并行"切输入维"。把它们前后配对，前者的输出切分正好是后者的输入切分
#   attention的o_proj：经过qkv_proj（列并行）+attention后，每张卡拿到的是 [N, num_heads/tp*head_dim]，是完整结果切开的一部分，天然适合作为行并行的输入
#   mlp的down_proj：经过gate_up_Proj（列并行）+silu*up后，每张卡拿到的是 [N, inter/tp]，是完整结果切开的一部分，天然适合作为行并行的输入 
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
        # bias是一维的[output_size]，输出维不切分，每个rank保留完整的bias
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        # 沿第1维，切出第tp_rank份：[output_size, input_size] -> [output_size, input_size/tp]
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # x [N, input_size/tp] # [output_size, input_size/tp]^T = [N, output_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rank0才加bias，保证一次计算只加一次bias，符合数学公式语义
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 各rank的部分和逐元素相加，结果原地写回y，所有rank得到相同的完整输入
            # all_reduce：必须所有rank都必须调用，否则会一直等待
            # rank0: y0 ─┐
            #            ├─ 求和 ─> 每个rank的y都变成 y0 + y1
            # rank1: y1 ─┘
            dist.all_reduce(y)
        return y
