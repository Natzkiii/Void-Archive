from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
from ChatData import ChatData
from torch.optim import AdamW
from torch.utils.data import DataLoader
import tqdm
import torch
from Setting import SETTINGS, Seed
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device Used To Train {device}")

tokenizer = GPT2Tokenizer.from_pretrained(SETTINGS['raw_model'])
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})


model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'])
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

chatData = ChatData(SETTINGS['Dataset_path'], tokenizer)
chatData =  DataLoader(chatData, SETTINGS['batch_size'])


start_time = time.time()



torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)


model_path = SETTINGS['model_path']

def train(chatData, model, optim):
    print("Loading Checkpoint")
    if(os.path.exists(model_path)):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    
    epochs = SETTINGS['epochs']
    loss_values = []
    try:
        for i in tqdm.tqdm(range(epochs)):
            epoch_loss = 0.0
            for X, a in chatData:
                X = X.to(device)
                a = a.to(device)
                optim.zero_grad()
                loss = model(X, attention_mask=a, labels=X).loss
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            epoch_loss /= len(chatData)
            loss_values.append(epoch_loss)
            print("Epoch {} loss: {:.4f}".format(i+1, epoch_loss))
    except:
        x = time.time() - start_time
        x = int(x)
        print("saving the model")
        torch.save(model.state_dict(), SETTINGS['saved_model_path'])
        print("Saved To " +  SETTINGS['saved_model_path'])
        print('Time Spend Training ', str(x), ' Seconds')
    
    



model.train()

#optim = AdamW(model.parameters())
optim = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)


train(chatData, model, optim)

# while True:
#   inp = input()
#   print(infer(inp))
# while True:
#   inp = input()
#   print(infer(inp))