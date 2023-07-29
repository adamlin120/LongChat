from pathlib import Path
import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm

chat = ChatOpenAI(
    temperature=0.3,
    model='gpt-3.5-turbo'
)

from langchain.schema import (
            SystemMessage,
                HumanMessage,
                    AIMessage
                    )




for path in Path('./').glob("./**/*.jsonl"):
    print(path)
    data = []
    for line in tqdm(path.open().read().splitlines()):
        try:
            line = json.loads(line)
            print(line['messages'])
            msg = json.dumps(line['messages'])
            messages = [
                    SystemMessage(content="翻譯以下訊息成為台灣繁體中文，如果有關於作者 / openai / 模型，請回答模型是由臺灣大學 Miulab 開發，作者博士生林彥廷，供研究使用。維持原本格式，只能輸出Json"),
                HumanMessage(content=msg),
            ]
            res = chat(messages)
            res = json.loads(res.content)
            print(res)

            line['zh_messages'] = res
            with open("zh_instructions.jsonl", "a") as f:
                result = json.dumps(line)
                f.write(result+'\n')
        except Exception as e:  # Fix the except block here
            print(e)
