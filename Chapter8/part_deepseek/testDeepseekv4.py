# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key='sk-1e18bf814204442c8d9f731af54289f5',
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "我是强化学习初学者，写一个rainbowdqn玩ataripong的python代码，使用pytorch、numpy库，一步一步来，注释尽可能详细，不要使用黑盒式函数"},
    ],
    stream=True,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

# 正确处理 Stream 对象
full_response = ""
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        full_response += content
        print(content, end="")  # 实时打印内容

# 如果你最终也需要完整内容，可以使用 full_response
print("\n完整响应内容为：")
print(full_response)