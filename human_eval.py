import transformers
import torch
from transformers import AutoModel, AutoTokenizer, set_seed, GPT2LMHeadModel, GPT2Tokenizer
from Setting import SETTINGS, Seed, inference
from colorama import Fore
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(SETTINGS['raw_model'])
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                              "sep_token": "<sep>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})
model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'], ignore_mismatched_sizes=True, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

def load(model):
    print("Loading Checkpoint")
    if os.path.exists(inference['inference_model_path']):
        checkpoint = torch.load(inference['inference_model_path'], map_location=torch.device(inference['inference_device']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
load(model)

def custom_seed():
     with torch.no_grad():
        set_seed(Seed['infer_seed'])
        torch.manual_seed(Seed['infer_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Seed['infer_seed'])
custom_seed()

def chatbot_response():
    while True:
        input_text = input(Fore.LIGHTGREEN_EX + 'USER:')
        input_text = input_text
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        generate = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            min_new_tokens=1,
            do_sample=True,
            num_beams=1,
            top_p=0.95,
            temperature=0.5,
            repetition_penalty=1.2,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        
        output_txt = tokenizer.decode(generate[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return output_txt

while True:
	print(Fore.LIGHTCYAN_EX + "Model: " + chatbot_response())
