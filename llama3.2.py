from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
base_model = "meta-llama/Llama-3.2-1B-Instruct"

backbone = AutoModelForCausalLM.from_pretrained(base_model) 

tokenizer = AutoTokenizer.from_pretrained(base_model)
label_str = "True"
label_index = tokenizer.encode(f" {label_str}")
print("Llama 3 Label Index", label_index)

if label_str not in tokenizer.get_vocab():
    tokenizer.add_tokens([label_str])
    tokenizer.save_pretrained("updated_llama_tokenizer")  # Optional, to reuse the tokenizer later

# Encode with the updated tokenizer
label_index = tokenizer.encode(label_str, add_special_tokens=False)
print("Updated Llama 3 Label Index:", label_index)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
label_index = tokenizer.encode(f" {label_str}")
print("GPT 2 Label Index", label_index)


