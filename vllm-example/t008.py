import os
import argparse
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "ai-app-changhe-cs-project"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "ls__1320d1c67bf8480097cb8c81462ca99f"

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

#DEFAULT_MODEL_NAME = "Qwen1.5-14B-Chat"
#OPEN_AI_KEY = "sk-PuEK85nbJC2zqiJsBbE56c60F65b47E899718fD62b02AdC2"
#OPENAI_API_BASE = "http://oneapi-uat.bytefinger.com/v1"

#OPEN_AI_KEY = "sk-wkEzlH4DN1Hani4J5dE759C154834e5bA69a79Cd8442D3Ef"
#OPENAI_API_BASE = "http://changhe-one-api.bytefinger.cn/v1"

DEFAULT_MODEL_NAME = "/workspace/qwen14b-bmodel/config"
OPEN_AI_KEY = "sk-wkEzlH4DN1Hani4J5dE759C154834e5bA69a79Cd8442D3Ef"
OPENAI_API_BASE = "http://localhost:8000/v1"

def main(args):
    is_stream = False
    if str(args.stream).lower() == 'true':
        is_stream = True
    else:
        is_stream = False

    print(f'Is use stream output: {is_stream}')

    model = ChatOpenAI(
        model_name=DEFAULT_MODEL_NAME,
        temperature=0,
        openai_api_key=OPEN_AI_KEY,
        openai_api_base=OPENAI_API_BASE,
        #max_tokens=4000
    )

    prompt = ChatPromptTemplate.from_template("{question}")

    chain = (
            prompt 
            | model.bind(stop=["<|im_start|>", "<|im_end|>", "<|endoftext|>", "!!!!"]) 
            | StrOutputParser()
    ).with_config(run_name="SampleCodeGenerateAnswer")

    questions = [
        # "出师表全文是什么？",
        # """
        # Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        # Role Description: 
        # Please reply in chinese.
        # Chat History:[HumanMessage(content='你是谁？'), AIMessage(content='我是来自中国的一名AI助手。')]
        # Follow Up Input: 最轻的元素是什么？
        # Standalone Question:
        # """,
        # "你好",
        # "你是谁？",
        "太阳从哪边升起来",
        "法国的首都在哪里？",
        "一年有多少天",
        "最轻的元素是什么？",
        "AI的未来在哪里",
        "先有鸡还是先有蛋",
        "生蚝煮熟了之后叫什么",
        "蛋白质由什么构成",
        "2的10次方等于",
        "离离原上草，",
        "水是剧毒的吗？",
        "Linux系统如何删除文件夹",
        "天为什么是蓝的",
        "圆周率和自然常数谁更大",
        "最轻的元素是什么",
        "闯红灯要判几年",
    ]

    output(chain, is_stream, questions)

def output(chain, is_stream, questions):
    for question in questions:
        print(f'question: {question}')
        if is_stream == False:
            result = chain.invoke({"question": question})
            print(f'answer: {result}\n')
        else:
            asyncio.run(stream_output(chain, question))

async def stream_output(chain, question: str):
    chunks = []
    print('answer: ', end="")
    async for chunk in chain.astream({"question": question}):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream",
        type=str,
        required=False
    )
    args = parser.parse_args()
    main(args)
