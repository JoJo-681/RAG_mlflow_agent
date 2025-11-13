"""OpenAI client for DeepSeek API integration."""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# ===============================
# 1. 加载环境变量
# ===============================
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

client = OpenAI(api_key=api_key, base_url=base_url)


# ===============================
# 2. 定义流式生成器
# ===============================
def stream_chat(prompt: str, model="deepseek-chat", temperature=0.7):
    """
    使用 yield 实现流式输出生成器 / Stream generator using yield:
    - 一边从 API 获取数据块 / Fetch data chunks from API
    - 一边逐步产出 (yield) 文本增量 / Yield text increments progressively
    - 不汇总，不 return / No aggregation, no return
    """
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Provide accurate and concise technical information based on the user's query.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta  # 每次产生一小段


# ===============================
# 3. 在终端逐步消费（模拟流式打印）
# ===============================
if __name__ == "__main__":
    USER_INPUT = "请帮我解释什么是RAG（检索增强生成）"
    print("\nAgent is thinking (streaming output):\n")
    for text_piece in stream_chat(USER_INPUT):
        sys.stdout.write(text_piece)
        sys.stdout.flush()
    print("\n\nOutput completed.\n")
