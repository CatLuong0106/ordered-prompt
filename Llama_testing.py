from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import pandas as pd


access_token = "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"
model_1 = "meta-llama/Llama-2-7b-chat-hf"
model_2="meta-llama/Llama-3.2-11B-Vision-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_1, token=access_token)

model_llama2 = AutoModelForCausalLM.from_pretrained(model_1, token=access_token)

model_llama3 = AutoModelForImageTextToText.from_pretrained(model_2, token=access_token)  

pipe = transformers.pipeline(
    "text-generation",
    model=model_llama2,
    torch_dtype=torch.float16,
    #device_map="auto",
    device=0,
    tokenizer=tokenizer)



# pipe_d=pipe(messages)
# print(pipe_d[0]['generated_text'][1]["content"])

df=pd.read_csv("prompts.csv")
print(df.columns)

print(df.iloc[0]["prompt"])

messages = [
    {"role": "user", "content": "Provided a set of sentences and the sentiment of the sentence, where 1 denotes positive and 0 denotes negative, classify the sentiment of the last sentence, where the sentiment is denoted by <mask>. Only provide 0 or 1 as output nothing else:"},
]
for i in range(len(df)):
    prompt=df.iloc[i]["prompt"]
    messages = [{"role": "user", "content": "Provided a set of sentences and the sentiment of the sentence, where 1 denotes positive and 0 denotes negative, classify the sentiment of the last sentence, where the sentiment is denoted by <mask>. Only print 0 or 1 as output nothing else. Do not say here's my output or anything else. Reply only in 1 or 0:"+prompt}]
    pipe_d=pipe(messages)
    print(pipe_d[0]['generated_text'][1]["content"])
    df['predicted_label'][i] = pipe_d[0]['generated_text'][1]["content"]
    print(i)

# Save to CSV (optional)
file_name="prompts_Llama-2-7b-chat-hf_predictions.csv"
df.to_csv(file_name, index=False)

