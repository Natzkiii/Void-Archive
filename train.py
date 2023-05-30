from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, set_seed
from colorama import Fore
from transformers.optimization import Adafactor, AdafactorSchedule
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from ChatData import ChatData
import tqdm, time, os
import math
from Setting import SETTINGS, Seed
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.utils").setLevel(logging.WARNING)
print(Fore.LIGHTRED_EX + "There Is One Error And One Warning Currently Unresolved" + Fore.LIGHTGREEN_EX + " [Last Updated: 2023/5/30]")

config_path ="config/config.json"
#config = GPT2Config.from_json_file(config_path)


device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
print(f"{Fore.LIGHTGREEN_EX}Device Using To Train {Fore.LIGHTRED_EX}{device}")


#tokenizer = GPT2Tokenizer.from_pretrained(SETTINGS['raw_model'], vocab_file)
tokenizer = GPT2Tokenizer.from_pretrained('tokenizer/')
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                              "sep_token": "<sep>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})

model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'], ignore_mismatched_sizes=True, config=config_path)
model.resize_token_embeddings(len(tokenizer))
model = torch.compile(model)
model = model.to(device)
chatData = ChatData(SETTINGS['Dataset_path'], tokenizer)
chatData =  DataLoader(chatData, SETTINGS['batch_size'], num_workers=4, shuffle=True, pin_memory=True)
model_path = SETTINGS['model_path']
epochs = SETTINGS['epochs']
#optim = AdamW(model.parameters())
optim = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.1,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)
scaler = torch.cuda.amp.GradScaler()
print(Fore.LIGHTGREEN_EX + "Initiating Model Training")

start_time = time.time()


torch.manual_seed(Seed["train_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Seed["train_seed"])


def load(model, optim):
    if os.path.exists(SETTINGS['model_path']):
        print(Fore.LIGHTGREEN_EX + "Loading Checkpoint")
        checkpoint = torch.load(SETTINGS['model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
        last_save = checkpoint['last_save']
    else:
        print(Fore.LIGHTRED_EX + "Error: Checkpoint Not Found Please Recheck The Checkpoint Path." + Fore.LIGHTGREEN_EX + "\nNo Worry's Now Starting Training From Scratch.")
load(model, optim)
params = list(model.parameters())
params = list(model.parameters())
num_params = sum([p.numel() for p in params])
print(Fore.LIGHTCYAN_EX + 'Number of parameters:', num_params)    

def calculate_perplexity(loss):
    try:
        if loss == float('inf'):
            return 1e+8
        else:
            return math.exp(loss)
    except OverflowError:
        return float('inf')


def train(chatData, model, optim):
#    print("Loading Checkpoint")
#    if(os.path.exists(model_path)):
#        model.load_state_dict(torch.load(model_path))
#        model.eval()
    loss_values = []
    perplexity_values = []
    last_save = 0
    try:
        with tqdm.tqdm(range(epochs),
          position=0, desc=Fore.LIGHTCYAN_EX + "epochs") as progBar1, tqdm.tqdm(range(len(chatData)),
            position=1, desc=Fore.LIGHTCYAN_EX + "training") as progBar0:
            for i in range(epochs):
                epoch_loss = 0.0
                for X, a in chatData:
                    X = X.to(device)
                    a = a.to(device)
                    with torch.cuda.amp.autocast():
                        output = model(X, attention_mask=a, labels=X)
                        loss = output.loss
                    #loss = torch.clamp(loss, -4.0, 4.0)
                    perplexity = calculate_perplexity(loss)
                    optim.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    epoch_loss += loss.item()
                    perplexity_values.append(perplexity)
                    progBar0.update(1)
                    progBar0.set_description(f"{Fore.LIGHTCYAN_EX}frag loss: {loss:.10f} | perplexity: {perplexity:.10f}")

                if i % SETTINGS['saveEvery'] == 0:
                    progBar1.set_description(f"{Fore.LIGHTCYAN_EX}Last Save: {last_save} {Fore.LIGHTGREEN_EX}Saving In Process ")
                    save_model = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'last_save': last_save,
                    }
                    torch.save(save_model, SETTINGS['saved_model_path'])
                    last_save = i
                    progBar1.set_description(f"{Fore.LIGHTCYAN_EX}Last Save: {last_save} |" + Fore.LIGHTGREEN_EX + "Saving Process Took [", str(x) + "] Seconds")

                epoch_loss /= len(chatData)
                loss_values.append(epoch_loss)
                perplexity_values.append(calculate_perplexity(epoch_loss))
                progBar0.reset()
                progBar1.update(1)
                progBar1.set_description(f"{Fore.LIGHTCYAN_EX}loss: {epoch_loss:.10f}, perplexity: {calculate_perplexity(epoch_loss):.10f}, {Fore.LIGHTGREEN_EX}Last Save: {last_save}")

    except KeyboardInterrupt:
        x = int(time.time() - start_time)
        print(Fore.LIGHTGREEN_EX + "saving the model")
        save_model = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'last_save': last_save,
                    }
        torch.save(save_model, SETTINGS['saved_model_path'])
        last_save = i
        print(Fore.LIGHTGREEN_EX + "Saved To [" +  SETTINGS['saved_model_path'] + "]")
        print(Fore.LIGHTGREEN_EX + 'Time Spend Training [', str(x), '] Seconds')
        exit(0)

    # Print final training loss and perplexity
    final_loss = loss_values[-1]
    final_perplexity = perplexity_values[-1]
    print("Training Loss:", final_loss)
    print("Training Perplexity:", final_perplexity)


train(chatData, model, optim)
