from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
from ChatData import ChatData
from torch.optim import AdamW
from Setting import SETTINGS, Seed
from torch.utils.data import DataLoader
import tqdm
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device Used To Inference {device}")

tokenizer = GPT2Tokenizer.from_pretrained(SETTINGS['raw_model'])
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})

model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'])
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)
np.random.seed(42)
randseed = np.random.randint(10000)

def load(model, optim):
    model.load_state_dict(torch.load(SETTINGS['inference_model_path']))
    model.eval()
    
chat_history = []

def infer(inp, chat_history):
    # Encode the input text
    torch.manual_seed(Seed['infer_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Seed['infer_seed'])

    
    input_text = " ".join(chat_history[-1000:]) # limit chat history to last 1000 messages
    inp = tokenizer(input_text + user, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    print(input_text)
    # Generate the output text
    # change temperature, num_beams to get better results
    output = model.generate(X, attention_mask=a, max_new_tokens=80, min_new_tokens=10, do_sample=True, num_beams=2, temperature=1.0, pad_token_id=tokenizer.eos_token_id, early_stopping=True)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the input text and generated text, and return only the generated text
    input_tokens = tokenizer.encode(inp["input_ids"].tolist()[0], skip_special_tokens=True)
    generated_tokens = tokenizer.encode(output)
    generated_text = tokenizer.decode(generated_tokens[len(input_tokens):], skip_special_tokens=True)
    chat_history.append(generated_text)
    return generated_text

#optim = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
optim = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)

print(optim)

load(model, optim)

while True:
    user = input("USER: ")
    if user == ("-end-"):
        print("GPT: Have A Nice Day")
        break
    chat_history.append(user)
    print("GPT:", infer(user, chat_history))
