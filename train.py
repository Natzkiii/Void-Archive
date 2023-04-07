from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from ChatData import ChatData
import tqdm, time, os

from Setting import SETTINGS, Seed

device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device Using To Train {device}")

print("Bing", end="\r")
tokenizer = GPT2Tokenizer.from_pretrained(SETTINGS['raw_model'])
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<sos>",
                                "eos_token": "<eos>"})
print("Bang", end="\r")
model = GPT2LMHeadModel.from_pretrained(SETTINGS['raw_model'])
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
chatData = ChatData(SETTINGS['Dataset_path'], tokenizer)
chatData =  DataLoader(chatData, SETTINGS['batch_size'])
model_path = SETTINGS['model_path']
epochs = SETTINGS['epochs']
print("Training..")

start_time = time.time()

torch.manual_seed(Seed["train_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Seed["train_seed"])



def train(chatData, model, optim):
    print("Loading Checkpoint")
    if(os.path.exists(model_path)):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    loss_values = []
    last_save = 0
    try:
        with tqdm.tqdm(range(epochs),
          position=0, desc="epochs", colour="green") as progBar1, tqdm.tqdm(range(len(chatData)),
            position=1, desc="training", colour="yellow") as progBar0:
            for i in range(epochs):
                epoch_loss = 0.0
                for X, a in chatData:
                    X = X.to(device)
                    a = a.to(device)
                    optim.zero_grad()
                    loss = model(X, attention_mask=a, labels=X).loss
                    loss.backward()
                    optim.step()
                    epoch_loss += loss.item()
                    progBar0.update(1)
                    progBar0.set_description(f"frag loss: {loss:.4f}")

                if( i % SETTINGS['saveEvery'] == 0):
                    progBar1.set_description(f"loss: {epoch_loss:.4f} LSA: {last_save} Saving: Started")
                    torch.save(model.state_dict(), SETTINGS['saved_model_path'])
                    last_save = i
                    progBar1.set_description(f"loss: {epoch_loss:.4f} LSA: {last_save} Saving: Done")

                epoch_loss /= len(chatData)
                loss_values.append(epoch_loss)
                progBar0.reset()
                progBar1.update(1)
                progBar1.set_description(f"loss: {epoch_loss:.4f} LSA: {last_save}")

            
                # print("Epoch {} loss: {:.4f}".format(i+1, epoch_loss))
    except KeyboardInterrupt:
        x = int(time.time() - start_time)
        print("saving the model")
        torch.save(model.state_dict(), SETTINGS['saved_model_path'])
        last_save = i
        print("Saved To " +  SETTINGS['saved_model_path'])
        print('Time Spend Training ', str(x), ' Seconds')
        exit(0)

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
print("saving the model")
torch.save(model.state_dict(), SETTINGS['saved_model_path'])
print("Done")