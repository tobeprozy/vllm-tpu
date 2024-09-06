python3 -m pip install dfss --upgrade 
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/vllm-tpu-v3.tar # 镜像
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/combine.tar.gz # qwen2-14B bmodel，如果自己编译，可以不下
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/config.tar.gz # 配置文件，可以不下，仓库已经包含了
python3 -m dfss --url=open@sophgo.com:/ezoo/vllm/qwen/driver-0619.tar.gz # 驱动，必须更新



