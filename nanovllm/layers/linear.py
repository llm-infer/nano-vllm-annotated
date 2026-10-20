import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


# 整除，并断言能整除（张量并行要求维度能被tp_size整除）
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# 张量并行（Tensor Parallelism）是什么？
#   模型太大或想更快时，把每一层的权重矩阵切成几份，分到多张GPU上，每张卡只算一部分，再合并结果
#
# 以 tp=2、MLP（H -> I -> H）为例，"列并行 + 行并行"的配对：
#
#             GPU 0                                   GPU 1
#   x [N, H]（两张卡都有完整的x）              x [N, H]
#      │                                        │
#      ▼ 列并行：W1的上半部分 [I/2, H]            ▼ 列并行：W1的下半部分 [I/2, H]
#   h0 [N, I/2]                               h1 [N, I/2]        <- 各自算出中间结果的一半，不需要通信
#      │                                        │
#      ▼ 行并行：W2的左半部分 [H, I/2]            ▼ 行并行：W2的右半部分 [H, I/2]
#   y0 [N, H]（部分和）                        y1 [N, H]（部分和）
#      │                                        │
#      └──────────── all_reduce：y = y0 + y1 ────┘                <- 只在这里通信一次
#      ▼                                        ▼
#   y [N, H]（两张卡都得到完整结果）            y [N, H]
#
# 为什么行并行的部分和相加等于完整结果？矩阵乘法按分块展开：
#   y = h @ W2^T = [h0, h1] @ [W2左, W2右]^T = h0 @ W2左^T + h1 @ W2右^T = y0 + y1
#
# 线性层基类，支持张量并行（Megatron-LM风格）
# 权重形状为[output_size, input_size]，与nn.Linear一致，计算 y = x @ W^T + b
#   x: [N, input_size] -> y: [N, output_size]，N为token数
# tp_dim：权重沿哪个维度切分，0为按输出维切（列并行），1为按输入维切（行并行），None为不切
# nn.Module：PyTorch中所有网络层/模型的基类
#   继承它之后，在__init__中赋值的 nn.Parameter 和子 nn.Module 会被自动登记，
#   model.parameters()、model.modules()、model.get_parameter(名字) 等都依赖这个登记
#   调用 layer(x) 时会自动执行 layer.forward(x)
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        # super().__init__()：调用父类（nn.Module）的初始化，必须在给self赋值参数之前调用
        super().__init__()
        # 权重切分的维度
        self.tp_dim = tp_dim
        # 当前rank编号，决定取完整权重中的第几份
        self.tp_rank = dist.get_rank()
        # 张量并行的总份数
        self.tp_size = dist.get_world_size()
        # 这里的input_size/output_size已经是切分后的本地大小
        # weight: [output_size, input_size]
        # torch.empty：分配显存但不初始化（值是随机的垃圾数据），因为马上会从checkpoint加载覆盖
        # nn.Parameter：把张量标记为"模型参数"，会被Module自动登记，名字就是属性名（如 "weight"）
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 挂上子类实现的weight_loader，load_model时调用它从完整权重中切出本rank的部分
        # Python对象可以随时添加新属性，这里给参数对象动态加了一个weight_loader属性
        # self.weight_loader 是"绑定方法"，调用时自动带上self，所以loader只需传 (param, loaded_weight)
        self.weight.weight_loader = self.weight_loader
        if bias:
            # bias: [output_size]
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 注册为None，保证self.bias属性存在
            self.register_parameter("bias", None)

    # 基类不实现forward，子类必须重写；直接调用会抛出NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 不切分的线性层，每个rank持有完整权重（本模型中未使用）
# weight: [output_size, input_size]
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

    # F.linear(x, W, b) = x @ W^T + b
    #   x [N, in] @ W^T [in, out] -> [N, out]，b为None时不加bias
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# 列并行：按输出维切分权重，每个rank计算输出的一部分
# 输入x是完整的，输出是切分的，不需要通信
#   完整权重：[output_size, input_size]
#   本地权重：[output_size/tp, input_size]
#   x: [N, input_size] -> y: [N, output_size/tp]
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
    # loaded_weight: [output_size, input_size] -> narrow -> [output_size/tp, input_size]
    # tensor.narrow(dim, start, length)：在第dim维上取 [start, start+length) 这一段，返回视图（不拷贝）
    #   例如 W 形状[4096, 1024]，tp=2，rank=1：W.narrow(0, 2048, 2048) 即 W[2048:4096, :]
    # param.data.copy_(src)：把src的数据原地拷贝进参数（形状必须一致）；.data 绕过参数的梯度追踪
    # （bias同理：[output_size] -> [output_size/tp]）
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # x: [N, input_size] -> [N, output_size/tp]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# 合并的列并行层：把多个输入相同的线性层（如MLP的gate_proj和up_proj）合并成一个，一次矩阵乘法完成
#   output_sizes = [I, I]（I = intermediate_size）
#   本地权重：[2*I/tp, H]，布局为 [gate的第rank份 (I/tp行); up的第rank份 (I/tp行)]
#   x: [N, H] -> y: [N, 2*I/tp]，前一半是gate、后一半是up，SiluAndMul直接chunk即可
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        # 各个子层的输出大小（切分前），如[intermediate_size, intermediate_size]
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # loaded_shard_id：当前加载的是第几个子层（0为gate，1为up）
    # loaded_weight：该子层的完整权重 [output_sizes[id], H]
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 该子层在本地权重中的起始行和行数（都是切分后的）
        # 例如I=3072，tp=2：gate写入本地第[0, 1536)行，up写入本地第[1536, 3072)行
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 本地权重中该子层对应的区域：[I/tp, H]
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 完整子层权重 [I, H] 沿输出维切成tp份，取本rank那份：[I/tp, H]
        # tensor.chunk(n, dim)：沿dim维平均切成n块，返回元组；[self.tp_rank] 取第rank块
        #   例如 [3072, 1024].chunk(2, 0) -> ([1536, 1024], [1536, 1024])
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# QKV合并的列并行层：把q_proj、k_proj、v_proj合并成一个
# 按注意力头切分：每个rank负责 Hq = total_heads/tp 个q头 和 Hkv = total_kv_heads/tp 个kv头
#   本地权重：[(Hq + 2*Hkv) * D, H]，布局为 [本rank的q头 (Hq*D行); k头 (Hkv*D行); v头 (Hkv*D行)]
#   x: [N, H] -> y: [N, (Hq + 2*Hkv) * D]
# 例如Qwen3-0.6B（16个q头，8个kv头，D=128，tp=1）：权重[(16+16)*128, 1024] = [4096, 1024]
#
# tp=2时，HF权重如何分配到各rank（Qwen3-0.6B：16个q头，8个kv头）：
#
#   HF checkpoint中的权重        切成2份          rank 0 本地qkv_proj.weight    rank 1 本地qkv_proj.weight
#   q_proj [16*128, H]   ──>  q头0~7 | q头8~15  ──>   第一段：q头 0~7              第一段：q头 8~15
#   k_proj [ 8*128, H]   ──>  k头0~3 | k头4~7   ──>   第二段：k头 0~3              第二段：k头 4~7
#   v_proj [ 8*128, H]   ──>  v头0~3 | v头4~7   ──>   第三段：v头 0~3              第三段：v头 4~7
#                                                    形状 [(8+4+4)*128, H]        形状 [(8+4+4)*128, H]
#   每个rank拿到"自己那几个头"的q、k、v，注意力可以在本卡上独立计算，不需要通信
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
        # 每个头的维度D
        self.head_size = head_size
        # 本rank的q头数Hq、kv头数Hkv
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 合并后的总输出维度（切分前）
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    # loaded_shard_id："q"、"k"、"v"，决定写入本地权重的哪一段
    # loaded_weight：q为[total_heads*D, H]，k/v为[total_kv_heads*D, H]
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 本地权重的行区间：q [0, Hq*D)，k [Hq*D, (Hq+Hkv)*D)，v [(Hq+Hkv)*D, (Hq+2*Hkv)*D)
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
        # 完整权重按头切成tp份，取本rank那份：[total*D, H] -> [total/tp*D, H]
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# 行并行：按输入维切分权重，每个rank用切分后的输入计算出完整形状的部分和，再all_reduce求和
#   完整权重：[output_size, input_size]
#   本地权重：[output_size, input_size/tp]
#   x: [N, input_size/tp] -> 部分和 [N, output_size] -> all_reduce -> [N, output_size]
# 例如：W[1024, 4096]，tp=2 -> 每个rank持有W[1024, 2048]，输入是列并行输出的那2048维
# 数学上：x @ W^T = Σ_rank x_rank @ W_rank^T，所以各rank部分和相加即为完整结果
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
        # bias是一维的 [output_size]，输出维不切分，每个rank都保存完整的bias
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        # 沿第1维（输入维）切出第tp_rank份：[output_size, input_size] -> [output_size, input_size/tp]
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # x: [N, input_size/tp] -> y: [N, output_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias只在rank 0加，否则all_reduce后bias会被加tp_size次
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 各rank的部分和逐元素相加，结果原地写回y，所有rank得到相同的完整输出
            # all_reduce：集合通信，所有rank都必须调用，否则会一直等待（卡死）
            #   rank0: y0 ─┐
            #              ├─ 求和 ─> 每个rank的y都变成 y0 + y1
            #   rank1: y1 ─┘
            dist.all_reduce(y)
        return y
