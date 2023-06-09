from transformers import AutoTokenizer, AutoModel, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
from colorama import Fore
from transformers.optimization import Adafactor, AdafactorSchedule
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from ChatData import ChatData
import tqdm, time, os
import math
import json
import pickle
from Setting import SETTINGS, Seed, Optimizer_Settings
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
print(Fore.LIGHTRED_EX + "There Might Be Unsolved Error's But Nothing Critical" + Fore.LIGHTGREEN_EX + "\n[Last Updated: 2023/6/8]")



device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
print(f"{Fore.LIGHTGREEN_EX}Device Using To Train {Fore.LIGHTRED_EX}{device}")

config_path ="config/config.json"


tokenizer = GPT2Tokenizer.from_pretrained(SETTINGS['raw_model'])
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                              "sep_token": "<sep>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})
model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'], ignore_mismatched_sizes=True, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
chatData = ChatData(SETTINGS['Dataset_path'], tokenizer)
chatData =  DataLoader(chatData, SETTINGS['batch_size'], num_workers=SETTINGS['num_workers'], shuffle=True, pin_memory=True)
model_path = SETTINGS['model_path']
epochs = SETTINGS['epochs']
start_time = time.time()


optimizer = Adafactor(
    model.parameters(),
    lr=Optimizer_Settings['lr'],
    eps=(1e-30, Optimizer_Settings['lr']),
    clip_threshold=Optimizer_Settings['grad_clip'],
    decay_rate=-Optimizer_Settings['decay_rate'],
    beta1=Optimizer_Settings['beta1'],
    weight_decay=Optimizer_Settings['weight_decay'],
    relative_step=Optimizer_Settings['relative_step'],
    scale_parameter=Optimizer_Settings['scale_parameter'],
    warmup_init=Optimizer_Settings['warmup_init'],
)

print(Fore.LIGHTGREEN_EX + "Initiating Model Training")


torch.manual_seed(Seed["train_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Seed["train_seed"])


def load(model, optimizer):
    if os.path.exists(SETTINGS['model_path']):
        print(Fore.LIGHTGREEN_EX + "Loading Checkpoint")
        checkpoint = torch.load(SETTINGS['model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(Fore.LIGHTRED_EX + "Error: Checkpoint Not Found Please Recheck The Checkpoint Path." + Fore.LIGHTGREEN_EX + "\nNo Worry's Now Starting Training From Scratch.")
load(model, optimizer)

# Print Parameters
params = list(model.parameters())
num_params = sum([p.numel() for p in params])
print(Fore.LIGHTCYAN_EX + 'Number of parameters:', num_params)
   

def train(chatData, model, optimizer):
    train_losses = []
    last_save = 0
    model.train()
    try:
        with tqdm.tqdm(range(epochs),
          position=0, desc=Fore.LIGHTCYAN_EX + "epochs") as progBar1, tqdm.tqdm(range(len(chatData)),
            position=1, desc=Fore.LIGHTCYAN_EX + "training") as progBar0:
            for i in range(epochs):
                for X, a in chatData:
                    X = X.to(device)
                    a = a.to(device)
                    output = model(X, attention_mask=a, labels=X)
                    loss = output[0]
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_losses.append(loss.detach())
                    ppl = torch.exp(loss.detach())
                    train_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
                    train_loss = np.mean(train_losses)
                    progBar0.update(1)
                    progBar0.set_description(f"{Fore.LIGHTCYAN_EX}Loss: {train_loss:.6f} | Perplexity: {ppl:.6f}")

                if i % SETTINGS['saveEvery'] == 0:
                    progBar1.set_description(f"{Fore.LIGHTCYAN_EX}Last Save: {last_save} {Fore.LIGHTGREEN_EX}Saving In Process ")
                    save_model = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    if not os.path.exists('saved_models/'):
                        os.makedirs('saved_models/')
                    else:
                        torch.save(save_model, SETTINGS['saved_model_path'])
                    last_save = i
                    x = int(time.time() - start_time)
                    progBar1.set_description(f"{Fore.LIGHTCYAN_EX}Last Save: {last_save} |" + Fore.LIGHTGREEN_EX + "Saving Process Took [", str(x) + "] Seconds")
                    
                progBar0.reset()
                progBar1.update(1)
                progBar1.set_description(f"{Fore.LIGHTCYAN_EX}Loss: {train_loss:.6f}, perplexity: {ppl:.6f}, {Fore.LIGHTGREEN_EX}Last Save: {last_save}")

    except KeyboardInterrupt:
        x = int(time.time() - start_time)
        print(Fore.LIGHTGREEN_EX + "saving the model")
        save_model = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
        torch.save(save_model, SETTINGS['saved_model_path'])
        last_save = i
        print(Fore.LIGHTGREEN_EX + "Saved To [" +  SETTINGS['saved_model_path'] + "]")
        print(Fore.LIGHTGREEN_EX + 'Time Spend Training [', str(x), '] Seconds')
        exit(0)


train(chatData, model, optimizer)