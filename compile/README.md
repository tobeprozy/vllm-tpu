# vllm框架模型导出和编译

## 目录

- [vllm框架模型导出和编译](#vllm框架模型导出和编译)
  - [目录](#目录)
  - [1 框架接入流程](#1-框架接入流程)
    - [1.1 基本流程](#11-基本流程)
    - [1.2 实现细节](#12-实现细节)
  - [2 模型导出与编译](#2-模型导出与编译)
    - [2.1 导出onnx](#21-导出onnx)
    - [2.2 编译bmodel](#22-编译bmodel)
  - [3 其他事项](#3-其他事项)
  - [4 具体步骤（直接开始）](#4-具体步骤直接开始)
    - [4.1 修改seq\_length](#41-修改seq_length)
    - [4.2 导出onnx](#42-导出onnx)
    - [4.3 导出bmodel](#43-导出bmodel)



## 1 框架接入流程

### 1.1 基本流程

tpu接入vllm框架流程与llm-tpu工程中的步骤略有不同，为了适配vllm框架中自带continuous batching以及paged attention等策略，需要按照vllm的kv\_cache存储方式来构造attention，以tpu kernel算子的形式实现。因此vllm中llm block部分将被分为三个部分，相当于llm-tpu工程中的block/block\_cache部分变为：pre attention 和post attention的bmodel，以及 attention算子。

此外，为了应用vllm框架中的采样策略，lm\_head部分只输出到logits，采样操作通过cpu实现，剩下的embedding部分与llm-tpu中一致。

### 1.2 实现细节

vllm中的kv\_cache为非连续存储，通过索引记录每一层cache的存储地址，因此attention部分需要通过独立的动态算子实现。

其余qkv的计算与mlp部分导出为bmodel，因此prefill/decode的bmodel输入输出都一致，只区分sequence length，目前是按照512长度作为prefill，通过分bin不同batch的长度作为decode (不同batch放在sequence 维度)。需要注意一点是attention算子不包含任何权重，因此完整attention部分的residual add和attention out matmul都是放在post bmodel 做的，因此多芯版本的post bmodel包含两次all-reduce，如下图。

![](image/image_cdwlcNOmkq.png)

## 2 模型导出与编译

### 2.1 导出onnx

导出onnx的过程与llm-tpu中主要区别在block，pre和post onnx都是block中的一部分，所以不能直接应用源码中的block导出，需要按照层级调用，手动将源码中layers的内部输出保存出来，pre onnx输入为input\_states和pos\_ids，计算到rotary\_pos\_emb后的qkv输出，post onnx输入为attn\_out和residual，需要先将attn\_out矩阵乘完再与residual相加输入到mlp内最后输出结果。

需要注意一点就是，最后需要实现动态长度，pre onnx的rotary pos embed部分需要将sin/cos的权重设为MAX\_POS\_EMB而不是输入sequence长度，这样pos ids才能支持更长序列。此外，lm\_head都不包括topk，只有rmsnorm+matmul。具体命令参考export_onnx.py
```python
class QwenPreAttnBlock(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(MAX_POS_EMB)
        self.cos_emb = self.rotary_emb[0].view(MAX_POS_EMB, HEAD_DIM)
        self.sin_emb = self.rotary_emb[1].view(MAX_POS_EMB, HEAD_DIM)

    def forward(self, hidden_states, position_ids):
        bsz, sqlen, _ = hidden_states.size()
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)

        layernorm_output = self.layer.ln_1(hidden_states)
        mixed_x_layer = self.layer.attn.c_attn(layernorm_output)

        query, key, value = mixed_x_layer.split(self.layer.attn.split_size, dim=2)
        query = query.view(bsz, sqlen, self.layer.attn.num_heads, self.layer.attn.head_dim)
        key = key.view(bsz, sqlen, self.layer.attn.num_heads, self.layer.attn.head_dim)
        value = value.view(bsz, sqlen, self.layer.attn.num_heads, self.layer.attn.head_dim)

        rotary_pos_emb = [cos_pos[:, -sqlen:, :, :], sin_pos[:, -sqlen:, :, :]]
        query = apply_rotary_pos_emb(query, rotary_pos_emb)
        key = apply_rotary_pos_emb(key, rotary_pos_emb)

        return query, key, value
```

```python
class QwenPostAttnBlock(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, attn_out, residual):
        attn_output = self.layer.attn.c_proj(attn_out)
        layernorm_input = attn_output + residual
        residual = layernorm_input
        layernorm_output = self.layer.ln_2(layernorm_input)
        mlp_output = self.layer.mlp(layernorm_output)
        hidden_states = residual + mlp_output
        return hidden_states
```

### 2.2 编译bmodel

具体编译命令可以参考compile.sh
以w8bf16举例：

```bash
qwen_block()
{
  for ((i=0; i<$layer; i++))
  do
    model_transform.py \
      --model_name pre_attn_block_${seqlen}seq_$i \
      --model_def ${pre_onnx_path} \
      --mlir pre_attn_block_${seqlen}seq_$i.mlir
    model_deploy.py \
      --mlir pre_attn_block_${seqlen}seq_$i.mlir \
      --quantize W8F16 \
      --chip bm1684x \
      --quant_input \
      --quant_output \
      --num_device $dev \
      --model pre_attn_block_${dev}dev_${seqlen}seq_${i}.bmodel

    model_transform.py \
      --model_name post_attn_block_${seqlen}seq_$i \
      --model_def ${post_onnx_path} \
      --mlir post_attn_block_${seqlen}seq_$i.mlir
    model_deploy.py \
      --mlir post_attn_block_${seqlen}seq_$i.mlir \
      --quantize W8F16 \
      --chip bm1684x \
      --quant_input \
      --quant_output \
      --num_device $dev \
      --model post_attn_block_${dev}dev_${seqlen}seq_${i}.bmodel
  done
}

embedding()
{
  model_transform.py \
      --model_name embedding \
      --input_shapes [[1,${seqlen}]] \
      --model_def ${emb_onnx_path} \
      --input_types 'int32' \
      --mlir embedding.mlir
  model_deploy.py \
      --mlir embedding.mlir \
      --quantize BF16 \
      --chip bm1684x \
      --quant_input \
      --quant_output \
      --num_device $dev \
      --model embedding_seq${seqlen}_${dev}dev.bmodel
}

lm_head()
{
  model_transform.py \
      --model_name lm_head \
      --model_def ${lm_head_path} \
      --input_shapes [[1,${seqlen},5120]] \
      --mlir lm_head.mlir
  model_deploy.py \
      --mlir lm_head.mlir \
      --quantize BF16 \
      --chip bm1684x \
      --quant_input \
      --quant_output \
      --num_device $dev \
      --model $lm_head_${seqlen}seq_${dev}dev.bmodel
}

```

## 3 其他事项

为了方便适配vllm的torch后端，整个流程使用了tpu-train的torch-tpu功能，以python的形式接入vllm，中间bmodel输入输出都以地址形式，通过torch申请device=tpu的输入输出tensor，再将输出输入tensor的地址输入到bmodel进行推理，再以tensor形式输入到attention算子，可以省去其中的搬运过程，但代价是不同框架的多次调用产生的latency以及cpu占用。



## 4 具体步骤（直接开始）

### 4.1 修改seq_length

切换到Qwen/compile/目录下后，手动将files/Qwen-14B-Chat/config.json中的`"seq_length": 512,`修改为你需要的长度
```shell
vi files/Qwen-14B-Chat/config.json
```

PS：
1. 由于导出的是静态onnx模型，所以必须手动修改为你所需要的长度

### 4.2 导出onnx

```shell
git clone https://huggingface.co/Qwen/Qwen-14B-Chat
cp files/Qwen-14B-Chat/* Qwen-14B-Chat 
export PYTHONPATH=Qwen-14B-Chat:$PYTHONPATH # export PYTHONPATH=/workspace/data4/zhiyuan.zhang/Qwen-14B-Chat:$PYTHONPATH
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

```shell
python export_onnx.py --model_path your_torch_path --seq_length 512 --batch_size 1
```

PS：
1. your_torch_path：从官网下载的或者自己训练的模型的路径，例如./Qwen-14B-Chat


### 4.3 导出bmodel

```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/VLLM/tpu-mlir_v1.9.beta.0-4-g0a5c44a32-20240905.tar.gz
tar -zxvf tpu-mlir_v1.9.beta.0-4-g0a5c44a32-20240905.tar.gz
cd tpu-mlir_v1.9.beta.0-4-g0a5c44a32-20240905
source envsetup.sh
```

以编译输入长度为512，量化为W8F16为例：
```shell
./compile.sh --name qwen-14b --seq_length 512 --quantize int8
```

PS：
1. mode：量化方式，目前支持int8/int4
2. name：模型名称，目前Qwen系列支持 qwen-1_8b/qwen-7b/qwen-14b/qwen-72b/
3. seq_length：模型支持的最大token长度