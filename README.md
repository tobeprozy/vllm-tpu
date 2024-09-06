# vllm-tpu

## 目录

- [vllm-tpu](#vllm-tpu)
  - [目录](#目录)
  - [简介](#简介)
  - [1 环境准备](#1-环境准备)
    - [1.1 依赖安装](#11-依赖安装)
    - [1.2 下载 vllm-TPU 的 docker 镜像和模型](#12-下载-vllm-tpu-的-docker-镜像和模型)
  - [2 docker部署](#2-docker部署)
    - [2.1 加载并启动docker](#21-加载并启动docker)
    - [2.2 准备配置文件和模型](#22-准备配置文件和模型)
  - [2.3 启动服务](#23-启动服务)
    - [2.3.1 离线服务](#231-离线服务)


## 简介
vLLM 是一个快速且易于使用的大模型推理加速工具。目前在 vLLM release 版本 0.3.3 的基础上完成了算能硬件产品 SC7 224T 的适配，可以不依赖 cuda 直接启动服务，对外接口协议与 vLLM 一致。当前支持的模型有 Llama2-13B 和 Qwen-14B

## 1 环境准备

### 1.1 依赖安装

```bash
sudo apt update 
sudo apt install python3-pip
```

### 1.2 下载 vllm-TPU 的 docker 镜像和模型

```bash
python3 -m pip install dfss --upgrade 
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/vllm-tpu-v3.tar # 镜像
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/combine.tar.gz # qwen2-14B bmodel，如果自己编译，可以不下
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/config.tar.gz # 配置文件，可以不下，仓库已经包含了
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/driver-0619.tar.gz # 驱动，必须更新
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/vllm-example.zip # vllm 示例代码，可以不下，使用本仓库代码
```

## 2 docker部署

### 2.1 加载并启动docker
```bash
docker load -i ./vllm-tpu-v3.tar
docker rm vllm-tpu -f 
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/docker_run.sh 
chmod +x docker_run.sh 
./docker_run.sh
```

### 2.2 准备配置文件和模型
```bash
tar zxvf combine.tar.gz 
mv combine qwen14b-bmodel 
tar zxvf config.tar.gz -C qwen14b-bmodel
```
需要在原有的 Qwen-14B 的 config 文件中加入以下参数：
a) 将 architectures 中的关键字改成 QWenSophgo，代表使用 Sophgo 的硬件进行推理:
```bash
 "architectures": [ 
 "QWenSophgo" 
 ],
 ```
b) core_num 参数，表示使用 Sophgo 产品的 8 个核进行推理，这个应该根据产品的型号进行
修改
```bash
"core_num": 8,
```

c) device_id 参数，可以通过 bm-smi 查看 Sophgo 产品在服务器中的每个核的序号
```bash
"device_id": [0,1,2,3,4,5,6,7], 
```
d) model_path 是使用 Sophgo 硬件推理所需要的模型格式 bmodel 位置，注意保证文件夹里 bmodel 的名字和json文件对应。
```bash
"model_path": "/workspace/qwen14b-bmodel", 
```
e) kv_block_num 是存储 kv cache 的 block 的数量
```bash
"kv_block_num":1000, 
```
f) prefill_bmodel_size 和 decode_bmodel_size 是和 bmodel 模型相关的固定参数，不需要修改
```
 "prefill_bmodel_size":512, 
 "decode_bmodel_size":16,
```

下载 20240715 版本的更新包，更新模型以及配置。（此版本相对于 0704 版降低了 CPU 占用率，从 36 个核降至 9 个核，目前占用 900%。）
```bash
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/update-0715.tar.gz 
tar -zxvf update-0715.tar.gz 
chmod +x update-0715/update.sh 
./update-0715/update.sh
```

## 2.3 启动服务
### 2.3.1 离线服务
注意修改model的路径。
```bash
cd vllm-example/
python3 offline_inference_sophgo.py --model /workspace/config_tobe
```