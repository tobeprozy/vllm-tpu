# vllm-tpu

## 目录

- [vllm-tpu](#vllm-tpu)
  - [目录](#目录)
  - [简介](#简介)
  - [1 环境准备](#1-环境准备)
    - [1.1 依赖安装](#11-依赖安装)
    - [1.2 下载docker镜像和模型](#12-下载docker镜像和模型)
    - [1.3 开启p2p（每次重启都需要开启）](#13-开启p2p每次重启都需要开启)
  - [2 docker部署](#2-docker部署)
    - [2.1 安装docker](#21-安装docker)
    - [2.2 加载并启动docker](#22-加载并启动docker)
    - [2.3 准备配置文件和模型](#23-准备配置文件和模型)
    - [2.4 更新libsophon和driver](#24-更新libsophon和driver)
  - [3 启动服务](#3-启动服务)
    - [3.1 离线服务](#31-离线服务)
  - [4 其它事项](#4-其它事项)
    - [4.1 vllm说明](#41-vllm说明)


## 简介
vLLM 是一个快速且易于使用的大模型推理加速工具。目前在 vLLM release 版本 0.3.3 的基础上完成了算能硬件产品 SC7 224T 的适配，可以不依赖 cuda 直接启动服务，对外接口协议与 vLLM 一致。当前支持的模型有 Llama2-13B 和 Qwen-14B

## 1 环境准备

### 1.1 依赖安装

```bash
sudo apt update 
sudo apt install python3-pip
```

### 1.2 下载docker镜像和模型

```bash
python3 -m pip install dfss --upgrade 
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/vllm-tpu-v3.tar # 镜像
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/combine.tar.gz # qwen2-14B bmodel，如果自己编译，可以不下
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/config.tar.gz # 配置文件，可以不下，仓库已经包含了
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/driver-0619.tar.gz # 驱动，必须更新
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/vllm-example.zip # vllm 示例代码，可以不下，使用本仓库代码
```


### 1.3 开启p2p（每次重启都需要开启）
查看 p2p 是否可用：
```bash
test_cdma_p2p 0x130000000 0 0x140000000 1 0x100000 
```

若显示带宽（Bandwidth）只有 1500MB/s 左右，可能是 p2p 不可用，需要按以下步骤开启：

a. iommu 没有关闭，按如下过程关闭：
```bash
sudo vi /etc/default/grub 
# 根据 CPU 类型选择添加 intel_iommu=off/amd_iommu=off 
# GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet intel_iommu=off" 
# GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amd_iommu=off" 
sudo update-grub 
sudo reboot
```

b. iommu 关闭后速度依然上不来，可能还需要配置一下 PCIE 链路。
```bash
lspci | grep 4052 
# 如果只有一张卡，显示可能如下，82 便是卡的编号。多张卡会显示多个，下面是三张8芯卡：
# 81:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:01.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:02.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:03.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:04.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:05.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:06.0 PCI bridge: PMC-Sierra Inc. Device 4052
# 82:07.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c1:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:01.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:02.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:03.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:04.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:05.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:06.0 PCI bridge: PMC-Sierra Inc. Device 4052
# c2:07.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e1:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:01.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:02.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:03.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:04.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:05.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:06.0 PCI bridge: PMC-Sierra Inc. Device 4052
# e2:07.0 PCI bridge: PMC-Sierra Inc. Device 4052
```

c. 配置 PCIE 链路，每张卡都需要运行如下命令，将配置卡号为82的所有芯片。
```bash
sudo setpci -v -s 82:*.0 ecap_acs+6.w=0 
```
然后再重新安装驱动即可。

## 2 docker部署

### 2.1 安装docker
若已安装docker，请跳过本节。 执行以下脚本安装 docker 并将当前用户加入 docker 组，获得 docker 执行权限。
```bash
# 安装docker
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker
# docker命令免root权限执行
# 创建docker用户组，若已有docker组会报错，可忽略
sudo groupadd docker
# 将当前用户加入docker组
sudo usermod -aG docker $USER
# 重启docker服务
sudo service docker restart
# 切换当前会话到新group或重新登录重启会话
newgrp docker​
```
### 2.2 加载并启动docker
```bash
docker load -i ./vllm-tpu-v3.tar
docker rm vllm-tpu -f 
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/docker_run.sh 
chmod +x docker_run.sh 
./docker_run.sh
```

### 2.3 准备配置文件和模型
```bash
tar zxvf combine.tar.gz 
mv combine qwen14b-bmodel 
```

需要在原有的 Qwen-14B 的 config 文件中加入以下参数：

a) 将 architectures 中的关键字改成 QWenSophgo，代表使用 Sophgo 的硬件进行推理，**一般不需要修改。**
```bash
 "architectures": [ 
 "QWenSophgo" 
 ],
 ```
b) core_num 参数，表示使用 Sophgo 产品的 8 个核进行推理，这个应该根据产品的型号进行，**一般不需要修改。**
修改
```bash
"core_num": 8,
```

c) device_id 参数，可以通过 bm-smi 查看 Sophgo 产品在服务器中的每个核的序号，**一般需要修改。**
```bash
"device_id": [0,1,2,3,4,5,6,7], 
```

d) model_path 是使用 Sophgo 硬件推理所需要的模型格式 bmodel 位置，**注意保证文件夹里 bmodel 的名字和json文件对应。**
```bash
"model_path": "/workspace/qwen14b-bmodel", 
```

e) kv_block_num 是存储 kv cache 的 block 的数量
```bash
"kv_block_num":1000, 
```

f) prefill_bmodel_size 和 decode_bmodel_size 是和 bmodel 模型相关的固定参数，**不需要修改**
```bash
 "prefill_bmodel_size":512, 
 "decode_bmodel_size":16,
```

### 2.4 更新libsophon和driver

```bash
sudo apt install dkms libncurses5 
tar zxvf driver-0619.tar.gz 
cd driver-0619 
sudo dpkg -i sophon-*.deb 
source /etc/profile 
```

下载 20240715 版本的更新包，更新模型以及配置。（此版本相对于 0704 版降低了 CPU 占用率，从 36 个核降至 9 个核，目前占用 900%。）
```bash
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/update-0715.tar.gz 
tar -zxvf update-0715.tar.gz 
chmod +x update-0715/update.sh 
./update-0715/update.sh
```

## 3 启动服务
### 3.1 离线服务
注意修改model的路径。
```bash
cd vllm-example/
python3 offline_inference_sophgo.py --model /workspace/config_tobe
```

## 4 其它事项
### 4.1 vllm说明

加载的容器默认安装了vllm，如果需要进行二次开发，可以在其安装路径下修改源文件：
```bash
cd /usr/local/lib/python3.10/dist-packages/vllm
```
