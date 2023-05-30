from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, set_seed
from transformers.optimization import Adafactor, AdafactorSchedule
from ChatData import ChatData
from colorama import Fore
from torch.optim import AdamW
from Setting import SETTINGS, Seed
from torch.utils.data import DataLoader
import tqdm
import os
import torch
import numpy as np
import re
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

config_path ="config/config.json"


device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
print(f"{Fore.GREEN}Device Being Use To Inference {Fore.LIGHTRED_EX}{device}")

tokenizer = GPT2Tokenizer.from_pretrained('tokenizer/')
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                              "sep_token": "<sep>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})

model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'], ignore_mismatched_sizes=True, config=config_path)
# model = torch.compile(model, mode="max-autotune")
model.resize_token_embeddings(len(tokenizer))
model = torch.compile(model)
model = model.to(device)


def load(model):
    print("Loading Checkpoint")
    if os.path.exists(SETTINGS['inference_model_path']):
        checkpoint = torch.load(SETTINGS['inference_model_path'], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        last_save = checkpoint['last_save']
        print(f"{Fore.GREEN}Model Is Trained For Toatal Of {Fore.LIGHTRED_EX}{last_save}")
load(model)
        
chat_history = []


def infer(inp, chat_history):
    # Encode the input text
 with torch.no_grad():
    set_seed(Seed['infer_seed'])
    torch.manual_seed(Seed['infer_seed'])
    if torch.cuda.is_available():
     torch.cuda.manual_seed_all(Seed['infer_seed'])
    
    max_chars = 400 # maximum number of characters to keep
    chat_history_str = "".join(chat_history) # join the chat history into a string
    input_text = chat_history_str[-max_chars:] # keep only the first `max_chars` characters

    inp = tokenizer(input_text + user.lower(), return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    print(input_text)
    # Generate the output text
    # change temperature, num_beams to get better results
    output = model.generate(X, attention_mask=a, max_new_tokens=100, min_new_tokens=1, do_sample=True, num_beams=1, top_p=0.95, temperature=0.85, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id, use_cache=True, early_stopping=True)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the input text and generated text, and return only the generated text
    input_tokens = tokenizer.encode(inp["input_ids"].tolist()[0], skip_special_tokens=True)
    generated_tokens = tokenizer.encode(output)
    generated_text = tokenizer.decode(generated_tokens[len(input_tokens):], skip_special_tokens=True)
    #remove white spaces and multiple tab
    generated_text = re.sub(r'[ \t]{2,}', ' ', generated_text)
    save_to_hist = "<sep>" + generated_text + "<sep>"
    chat_history.append(save_to_hist)
    return generated_text


while True:
    user = input("USER:")
    if user == ("-end-"):
        print("GPT: Have A Nice Day")
        break
    chat_history.append(user)
    print("GPT:", infer(user, chat_history))
