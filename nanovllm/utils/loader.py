import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


# 默认的权重加载方式：整体拷贝，适用于没有切分的参数（如RMSNorm的weight [H]）
# param：模型中的参数；loaded_weight：checkpoint中的权重，两者形状必须一致
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# 从模型目录加载safetensors权重
# model：已构建好结构的模型（参数是空张量）；path：模型目录，包含一个或多个*.safetensors文件
# 两个特殊处理：
#   1. 合并参数：HF权重中q_proj/k_proj/v_proj是三个独立矩阵，这里合并成一个qkv_proj（gate/up同理），
#      一次矩阵乘法代替多次，需要把每个分量写到合并参数的对应区域
#   2. 张量并行切分：每个参数有自己的weight_loader，负责从完整权重中切出当前rank需要的部分
#      例如tp=2时，q_proj.weight [Hq_total*D, H] 只取一半行写入本rank的qkv_proj.weight
def load_model(model: nn.Module, path: str):
    # 例如 {"q_proj": ("qkv_proj", "q"), ...}：HF权重名中的片段 -> (合并后的参数名片段, 分片id)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    #
    # 加载流程：
    #   遍历每个 *.safetensors 文件
    #     遍历文件中的每个权重名 weight_name
    #       名字中包含 q_proj/k_proj/v_proj/gate_proj/up_proj？
    #         是 -> 换成合并后的参数名，调用该参数的weight_loader(param, 权重, shard_id)，写入对应区域并切分
    #         否 -> 直接按名字找参数，调用weight_loader(param, 权重)（没有自定义loader就整体拷贝）
    #
    # os.path.join：拼接路径，自动处理 "/"
    # glob(模式)：返回匹配通配符的文件路径列表，"*.safetensors" 匹配所有以.safetensors结尾的文件
    for file in glob(os.path.join(path, "*.safetensors")):
        # safe_open按需读取（内存映射），不会一次性把整个文件读进内存
        # with语句：进入时打开文件，离开代码块时自动关闭（即使中途出错也会关闭）
        # "pt"：返回PyTorch张量；"cpu"：读到CPU内存
        with safe_open(file, "pt", "cpu") as f:
            # weight_name：checkpoint中的参数名，如 "model.layers.0.mlp.gate_proj.weight"
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 例如：model.layers.0.self_attn.q_proj.weight -> model.layers.0.self_attn.qkv_proj.weight，shard_id="q"
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        # model.get_parameter("a.b.c")：按带点的完整名字找到参数对象
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # f.get_tensor：从文件读出完整权重（CPU），weight_loader切分后拷贝到GPU参数中
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                # for...else：没有匹配到合并参数（没有break）时执行，按普通参数加载
                #   for循环正常走完（没有break）才执行else；中途break则跳过else
                #   这里用来表达"在映射表里找了一圈都没找到"的情况
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
