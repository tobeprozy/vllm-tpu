import argparse
import asyncio
from typing import AsyncGenerator, List, Tuple
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    pbar = tqdm(total=len(input_requests))

    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request, 10, 512
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    # outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    i = 0
    for task in tasks:
        outputs = await task
        print((1+i),'input:  ', input_requests[i],'\n')
        print((1+i),'output:', outputs.generated_text,'\n')
        i += 1

    return outputs


def main(args: argparse.Namespace):
    model_id = args.model

    prompts = [
        # "出师表全文是什么？",
        # "<|im_start|>user\n出师表全文是什么？<|im_end|>\n<|im_start|>assistant\n",
        # "<|im_start|>user\n天为什么是蓝的<|im_end|>\n<|im_start|>assistant\n",
        # "天为什么是蓝色的",
        # "<|im_start|>user\n天为什么是蓝色的<|im_end|>\n<|im_start|>assistant\n"
        # "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        # "<|im_start|>user\nGiven the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\n\nRole Description:\nPlease reply in chinese:Chinese\nChat History:[HumanMessage(content='你好'), AIMessage(content='你好，很高兴为您服务。请问有什么可以帮助您的？')]\nFollow Up Input: 你是谁\nStandalone Question:<|im_end|>\n<|im_start|>assistant\n"
        "Hellow bro的中文是什么",
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
        "离离原上草，"
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
    
    asyncio.run(
        benchmark(
            backend='vllm',
            api_url="http://localhost:8000/v1/completions",
            model_id=model_id,
            input_requests=prompts,
            request_rate=float("inf"),
        ))

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
