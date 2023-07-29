import json
import pandas as pd
from tqdm import tqdm

from lang



from datasets import load_dataset

dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split='train')

chat = ChatOpenAI(temperature=0.3, model='gpt-3.5-turbo')
current = pd.read_json("zh_instructions.evol.jsonl", lines=True)
cur_ids = set(current['id'].unique())

model_config = LMConfig(provider="openai_chat", model="gpt-3.5-turbo-16k")
dataset = _contexts_to_prompts(
    full_contexts=dataset,
    prompt_template=dataset,
    model_config=model_config,
    context_length=15000,
)
generate_from_chat_prompt(
    model_config=model_config,
    temperature=0.7,
    max_tokens=8000,
    top_p=0.9,
    context_length=15000,
    requests_per_minute=3500,
)

for line in tqdm(dataset):
    if line['idx'] in cur_ids:
        continue
    msg = line['conversations']
    instruct_turn = ChatTurn(
        role="system",
        content="翻譯以下訊息成為台灣繁體中文，如果有關於作者 / openai / 模型，請回答模型是由臺灣大學 Miulab 開發，作者博士生林彥廷，供研究使用。維持原本格式，只能輸出Json",
    )
    content_turn = ChatTurn(
        role="user",
        content=json.dumps(msg),
    )
    chat_messages = ChatMessages(
        turns=[instruct_turn, content_turn],


