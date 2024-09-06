from vllm import LLM, SamplingParams
import sys
sys.path.append(".")
from transformers import AutoModel
import argparse


def main(args: argparse.Namespace):
    model_id = args.model
    # Sample prompts.
    prompts = [
        "你好",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is"
        "太阳从哪边升起来",
        "水是剧毒的吗？",
        "法国的首都在哪里？",
        "一年有多少天",
        "最轻的元素是什么",
        "AI的未来在哪里",
        "先有鸡还是先有蛋",
        "生蚝煮熟了之后叫什么",
        "闯红灯要判几年",
        "天为什么是蓝的",
        "蛋白质由什么构成",
        "Linux系统如何删除文件夹",
        "圆周率和自然常数谁更大",
        "2的10次方等于",
        "离离原上草，",
        # "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
        # "Hello, who are you, and where are you from?",
        # "Who is the president of the United States?",
        # "The capital of France is",
        # "The future of AI is",
        # "How might climate change influence migration patterns across different continents in the future?",
        # "Could you describe the potential impact of quantum computing on global encryption standards?",
        # "Can you explain how machine learning differs from traditional programming in detail?",
        # "What are the two basic postulates of special relativity?",
        # "When did dinosaurs become extinct?",
        # "Which came first, the chicken or the egg?",
        # "How is machine learning being used to predict and manage natural disasters?",
        # "In what ways can virtual reality technology transform education and training?",
        # "What strategies are most effective in conserving biodiversity in urban environments?",
        # "How do advancements in material science impact the development of sustainable packaging?",
        # "What are the ethical considerations of gene editing technologies like CRISPR?",
        # "How does the gut microbiome influence human health and disease?"

    ]
    
    # Create a sampling params object.
    sampling_params = SamplingParams(
            temperature=0.0, 
            stop=['<|im_end|>', '<|im_start|>', '<|endoftext|>'], 
            top_p=0.75)
    
    # Create an LLM.
    llm = LLM(model=model_id, trust_remote_code=True) 
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    args = parser.parse_args()
    main(args)

