from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, set_seed
from Setting import SETTINGS
import torch
import numpy as np
import re
import argparse
import sys
import json

config_path ="m/config.json"

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def load():
    model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'])
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.load_state_dict(torch.load(SETTINGS['inference_model_path']))
    model.eval()
    return model

def rndSeed():
    seed = np.random.randint(10000, 99999)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
    return seed

def infer(inp):
 with torch.no_grad():
    inp = tokenizer(inp.lower(), return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, max_new_tokens=50, min_new_tokens=1, do_sample=True,
                             num_beams=1, temperature=1.0, repetition_penalty=1.0, 
                             pad_token_id=tokenizer.eos_token_id, early_stopping=True)
    
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    input_tokens = tokenizer.encode(inp["input_ids"].tolist()[0], skip_special_tokens=True)
    generated_tokens = tokenizer.encode(output)
    generated_text = tokenizer.decode(generated_tokens[len(input_tokens):], skip_special_tokens=True)
    generated_text = re.sub(r'[ \t]{2,}', ' ', generated_text)
    return generated_text

words = []
config = GPT2Config.from_json_file(config_path)

device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
print(colored(255, 105, 245, f"Using {device} to inference"))

tokenizer = GPT2Tokenizer.from_pretrained('tokenizer/')
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})

if(len(sys.argv) >= 2):
    print(colored(255, 105, 245, "initiating..."))
    model = load()
    if(len(sys.argv) >= 3):
        msg = sys.argv[2]
    else:
        msg = input(colored(42, 150, 245, "user: "))
    for i in range(int(sys.argv[1])):
        s = rndSeed()
        res = infer(msg)
        print(colored(200, 144, 99, "seed:"), colored(115, 255, 0, s), colored(252, 255, 105, res))
        words.extend(res.split())
    print(colored(177, 255, 105, f"from {len(words)} words found {len(set(words))} unique words."))

    fq = {item:words.count(item) for item in words}
    sorted = sorted(fq.items(), key=lambda x: x[1], reverse=True)
    for i, (k, v) in enumerate(sorted[:5]):
        print(f"{colored(252, 255, 105, k)} \t:\t {colored(115, 255, 0, v)}")

else:
    print(colored(240, 233, 110, sys.argv[0]), colored(255, 105, 245, "<NumberOfRounds> [<Message>]"))
