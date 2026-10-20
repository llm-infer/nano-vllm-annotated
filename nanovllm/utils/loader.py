import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


# 默认的权重加载方式：整体拷贝，适用于没有切分的参数（如RMSNorm的weight）
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# 从模型目录加载safetensors权重
# 两个特殊处理：
#   1. 合并参数：HF权重中q_proj/k_proj/v_proj是三个独立矩阵，这里合并成一个qkv_proj（gate/up同理），
#      一次矩阵乘法代替多次，需要把每个分量写到合并参数的对应区域
#   2. 张量并行切分：每个参数有自己的weight_loader，负责从完整权重中切出当前rank需要的部分
def load_model(model: nn.Module, path: str):
    # 例如 {"q_proj": ("qkv_proj", "q"), ...}：HF权重名中的片段 -> (合并后的参数名片段, 分片id)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        # safe_open按需读取，不会一次性把整个文件读进内存
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 例如：model.layers.0.self_attn.q_proj.weight -> model.layers.0.self_attn.qkv_proj.weight，shard_id="q"
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                # for...else：没有匹配到合并参数（没有break）时执行，按普通参数加载
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
