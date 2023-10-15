from transformers import OPTForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor
import torch
import os
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)




class Void_Trainer():
    def __init__(self, gpu_id : int, num_epoch : int, dlocation : str) -> None:
        self.epoch = num_epoch
        self.gpu_id = gpu_id
        self.location = dlocation
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
        self.model = OPTForCausalLM.from_pretrained('facebook/opt-350m')
        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.Dataset = CustomDataset(self.location, self.tokenizer)
        self.DataLoader = DataLoader(self.Dataset, batch_size=2, sampler=DistributedSampler(self.Dataset), pin_memory=True, num_workers=2, shuffle=False)
        self.optimizer = Adafactor(self.model.parameters(), lr=5e-5, decay_rate=-1.0, clip_threshold=1.0, relative_step=False, warmup_init=False, scale_parameter=True)
        
    def train(self):
        for i in range(self.epoch):
            for batch_idx, batch in enumerate(self.DataLoader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.gpu_id)
                attention_mask = attention_mask.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                forward = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = forward[0]
                loss.backward()
                print(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                
                
                

def main(rank : int, world_size : int, num_epoch : int, dlocation: str):
    ddp_setup(rank, world_size)
    void_trainer = Void_Trainer(rank, num_epoch, dlocation)
    void_trainer.train()
    destroy_process_group()
    
    
    

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    num_epoch = 89
    dlocation = '/kaggle/working/Void_Trainer/data.db'
    mp.spawn(main, args=(world_size, num_epoch, dlocation), nprocs=world_size)