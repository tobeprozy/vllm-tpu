
#!/bin/bash
set -ex
models=
folder="tmp"
num_device=8
mode_args=""
name=""
seq_length=
num_layers=
batch_size=1
combine_dir=
bmodel_list=""
combine_model=

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    --num_device)
        num_device="$2"
        shift 2
        ;;
    --name)
        name="$2"
        shift 2
        ;;
    --quantize)
        quantize="$2"
        shift 2
        ;;
    *)
        echo "Invalid option: $key" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    esac
done

if [[ -z "$quantize" ]]; then
    echo "Error: --quantize is required: int4, int8" >&2
    exit 1
fi

if [ "$quantize" = "int4" ]; then
  type="W4F16"
  echo "quantize type: W4F16"
elif [ "$quantize" = "int8" ]; then 
  type="W8F16"
  echo "quantize type: W8F16"
else
  >&2 echo -e "Error: Invalid quantize $quantize"
  exit 1
fi


if [ "$name" = "qwen-1_8b" ]; then
  num_layers=24
  hidden_size=2048
  echo "Compile Qwen-1_8B"
elif [ "$name" = "qwen-7b" ]; then 
  num_layers=32
  hidden_size=4096
  echo "Compile Qwen-7B"
elif [ "$name" = "qwen-14b" ]; then
  num_layers=40
  hidden_size=5120
  echo "Compile Qwen-14B"
elif [ "$name" = "qwen-72b" ]; then
  num_layers=80
  hidden_size=8192
  echo "Compile Qwen-72B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen-1_8b|qwen-7b|qwen-14b\033[0m"
  exit 1
fi

embedding()
{
  local dev=$1
  local seqlen=$2
  model_transform.py \
      --model_name embedding \
      --input_shapes [[1,${seqlen}]] \
      --model_def ../tmp/onnx/embedding.pt \
      --input_types 'int32' \
      --mlir embedding.mlir

  model_deploy.py \
      --mlir embedding.mlir \
      --quantize F16 \
      --chip bm1684x \
      --quant_output \
      --num_device ${dev} \
      --model embedding_seq${seqlen}_${dev}dev.bmodel
  bmodel_list+=" embedding_seq${seqlen}_${dev}dev.bmodel"
  mv *.bmodel ../${combine_dir}
  rm ./* -r
}

lm_head()
{
  local dev=$1
  local seqlen=$2
  model_transform.py \
      --model_name lm_head \
      --model_def ../tmp/onnx/lm_head.pt \
      --input_shapes [[1,${seqlen},${hidden_size}]] \
      --mlir lm_head.mlir

  model_deploy.py \
      --mlir lm_head.mlir \
      --quantize F16 \
      --chip bm1684x \
      --quant_input \
      --quant_output \
      --num_device $dev \
      --model lm_head_${seqlen}seq_${dev}dev.bmodel
  bmodel_list+=" lm_head_${seqlen}seq_${dev}dev.bmodel"
  mv *.bmodel ../${combine_dir}
  rm ./* -r
}

pre_block()
{
  local dev=$1
  local layer=$2
  local seq_length=$3
  for ((i=0; i<$layer; i++))
  do
  model_transform.py \
    --model_name pre_attn_block_${seq_length}seq_$i \
    --input_shapes "[[1,$seq_length,$hidden_size], [1,$seq_length]]" \
    --model_def ../tmp/onnx/pre_attn_block_dyn_seq_${i}.onnx \
    --mlir pre_attn_block_${seq_length}seq_$i.mlir

  model_deploy.py \
    --mlir pre_attn_block_${seq_length}seq_$i.mlir \
    --quantize $type \
    --chip bm1684x \
    --quant_input \
    --quant_output \
    --num_device $dev \
    --model pre_attn_block_${dev}dev_${seq_length}seq_${i}.bmodel
  bmodel_list+=" pre_attn_block_${dev}dev_${seq_length}seq_${i}.bmodel"
  done
  mv *.bmodel ../${combine_dir}
  rm ./* -r
}

post_block()
{
  local dev=$1
  local layer=$2
  local seq_length=$3
  for ((i=0; i<$layer; i++))
  do
  model_transform.py \
    --model_name post_attn_block_${seq_length}seq_$i \
    --input_shapes "[[1,$seq_length,$hidden_size], [1,$seq_length,$hidden_size]]" \
    --model_def ../tmp/onnx/post_attn_block_dyn_seq_${i}.onnx \
    --mlir post_attn_block_${seq_length}seq_$i.mlir

  model_deploy.py \
    --mlir post_attn_block_${seq_length}seq_$i.mlir \
    --quantize $type \
    --chip bm1684x \
    --quant_input \
    --quant_output \
    --num_device $dev \
    --model post_attn_block_${dev}dev_${seq_length}seq_${i}.bmodel
  bmodel_list+=" post_attn_block_${dev}dev_${seq_length}seq_${i}.bmodel"
  done
  mv *.bmodel ../${combine_dir}
  rm ./* -r
}

# ./compile_1.sh --name qwen-14b --seq_length 512
combine_model=$name"_"$model_$num_device"dev.bmodel"
combine_dir=combine/
mkdir -p $combine_dir
echo $combine_dir
mkdir -p gen

pushd gen
# embedding
embedding $num_device 512
# prefill=512
pre_block $num_device $num_layers 512
post_block $num_device $num_layers 512
# decode=1,2,4,8,16
pre_block $num_device $num_layers 1
post_block $num_device $num_layers 1
pre_block $num_device $num_layers 2
post_block $num_device $num_layers 2
pre_block $num_device $num_layers 4
post_block $num_device $num_layers 4
pre_block $num_device $num_layers 8
post_block $num_device $num_layers 8
pre_block $num_device $num_layers 16
post_block $num_device $num_layers 16
# lm_head
lm_head 1 16
popd

pushd $combine_dir
tpu_model --combine $bmodel_list -o ../$combine_model
popd
