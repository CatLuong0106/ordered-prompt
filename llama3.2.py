from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
base_model = "meta-llama/Llama-3.2-1B-Instruct"

backbone = LlamaForCausalLM.from_pretrained(base_model) 
tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)