import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


# 默认权重加载方式，整体拷贝，适用于没有切分的参数
# param：模型中的参数；loaded_weight：从权重文件中加载的张量，两者必须形状一致
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# .safetensors：名字->张量的字典，只有数据
# model：按config搭好的空架子，只有结构
# load_model：按照名字把张量填进model里同名的参数
#
# 1）模型目录里有什么
#   /Qwen3-0.6B
#     ├── config.json          -> hf_config: 层数、头数、hidden_size等等
#     ├── tokenizer.json       -> 分词器
#     ├── model.safetensors    -> 权重数据，大模型会拆成多个文件，如model-001-of-004.safetensors
# 
# 2）两边分别长什么样
#                                                              Qwen3ForCausalLM
#   .safetensors 文件（字典：名字 -> 张量）                          ├── model: Qwen3Model
#   model.embed_tokens.weights               [151936, 1024]      │   ├── embed_tokens               .weight
#                                                                │   ├── layers（ModuleList，28个）
#                                                                │   │   └── [0]: Qwen3DecoderLayer
#   model.layers.0.input_layernorm.weight    [1024]              │   │       ├── input_layernorm    .weight
#                                                                │   │       ├── self_attn
#   model.layers.0.self_attn.q_proj.weight   [2048, 1024]        │   │       │   ├── qkv_proj       .weight  <- q/k/v三个合成一个
#   model.layers.0.self_attn.k_proj.weight   [1024, 1024]        │   │       │   ├── q_norm/k_norm  .weight 
#   model.layers.0.self_attn.v_proj.weight   [1024, 1024]        │   │       │   └── o_proj         .weight
#   model.layers.0.self_attn.q_norm.weight   [128]               │   │       ├── post_attention_layernorm .weight
#   model.layers.0.self_attn.o_proj.weight   [1024, 2048]        │   │       └── mlp            
#   model.layers.0.mlp.gate_proj.weight      [3072, 1024]        │   │            ├── gate_up_proj  .weight  <- gate/up两个合成一个
#   model.layers.0.mlp.up_proj.weight        [3072, 1024]        │   │            └── down_proj     .weight
#   model.layers.0.mlp.down_proj.weight      [1024, 3072]        │   └── norm       .weight       
#   ...(第1~27层同理)                                             └── lm_head         .weight  <- 与embed_tokens共享
#   model.norm.weight                        [1024]
# 
# 3）参数名字从哪来：从根往下，把属性名用点连起来，ModuleList用下标
#       model.layers[0].self_attn.o_proj.weight  ->  model.layers.0.self_attn.o_proj.weight
# 
# 4）名字对不上的有5种：q_proj、k_proj、v_proj、gate_proj、up_proj
#    因为nano-vllm把它们合并成了qkv_proj、gate_up_proj（一次矩阵乘法代替多次）
#    所以先用packed_modules_mapping改名，再告诉weight_loader这是q/k/v中的哪一份，写到合并参数的对应行
#
# 5）加载顺序
#    Qwen3ForCasualLM           ->    按config建好所有层，参数已在GPU上，形状正确，但只是占位，值没有意义（linear是random，RMSNorm全是1）
#    load_model(model, path)    ->    逐个读文件里的张量（CPU），再拷贝到对应的张量里（GPU），空架子变成能用的模型
#   model里不需要从权重文件中加载的东西：RoPE的cos_sin_cache（启动时算），KV cache（启动后分配）  
def load_model(model: nn.Module, path: str):
    # packed_modules_mapping = {
    #     "q_proj": ("qkv_proj", "q"),
    #     "k_proj": ("qkv_proj", "k"),
    #     "v_proj": ("qkv_proj", "v"),
    #     "gate_proj": ("gate_up_proj", 0),
    #     "up_proj": ("gate_up_proj", 1),
    # }
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 加载流程：TP>1时，每个weight_loader只取属于自己的那一部分写入
    #     checkpoint中的权重名           形状              命中映射？    写到哪里、谁来写
    #     ────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #     ...self_attn.q_proj.weight   [2048, 1024]      q_proj      qkv_proj.weight 第0~2047行
    #     ...self_attn.k_proj.weight   [1024, 1024]      k_proj      qkv_proj.weight 第2048~3071行
    #     ...self_attn.v_proj.weight   [1024, 1024]      v_proj      qkv_proj.weight 第 3072~4095 行
    #     ...mlp.gate_proj.weight      [3072, 1024]      gate_proj   gate_up_proj.weight 第0~3071行
    #     ...mlp.up_proj.weight        [3072, 1024]      up_proj     gate_up_proj.weight 第 3072~6143 行
    #     ...self_attn.o_proj.weight   [1024, 2048]        否         同名参数，RowParallelLinear 的 loader
    #     ...input_layernorm.weight    [1024]              否         同名参数，default_weight_loader 整体拷贝
    #     model.embed_tokens.weight    [151936, 1024]      否         同名参数，VocabParallelEmbedding 的 loader
    for file in glob(os.path.join(path, "*.safetensors")):
        # safe_open：按需读取（内存映射），不会一次性把整个文件读到内存
        # pt：返回Pytorch张量
        # cpu：读到CPU
        with safe_open(file, "pt", "cpu") as f:
            # weight_name：checkpoint的参数名，比如model.layers.0.mlp.gate_proj.weight
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 合并到哪，如何合并
                        # ...self_attn.q_proj.weight -> self_attn.qkv_proj.weight, shared_id=q
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        # model.get_parameter("a.b.c")：按照带点的完整名字找到参数对象
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # f.get_tensor：从文件中读出权重，weight_loader切分后拷贝到GPU参数中
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
