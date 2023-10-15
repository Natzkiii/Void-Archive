from torch.utils.data import DataLoader
from Void_Workers.model_manager import Model_Manager
from Void_Workers.Void_Dispatcher import CustomDataset
from Void_Workers import optimizer_manager
from Void_Workers.Void_Preserver import Model_State_Preserver
from Void_Workers.Void_Card import TextGenModelCard
from Void_Workers import Void_Logger
from torch.nn.utils import clip_grad_value_
from transformers.modeling_utils import load_sharded_checkpoint
import torch
import numpy as np
import sys
import os
from colorama import Fore, Style
from tqdm import tqdm
import lightning as Light
import datetime


def setup_fabric():
    global fabric
    fabric = Light.Fabric(accelerator='auto', devices='auto', strategy='auto', precision='32')
    fabric.launch()
    

class Void_Trainer():
    def __init__(self, compute_type : str, model_type : str, tokenizer_name : str, model_name : str, optimizer_config : None, lora_config : None, dataset : None, dataset_path : str, dataset_max_line_length : int, batch_size : int, num_workers : int, shuffle : bool,  total_epochs : int, gradient_checkpointing : bool, shard_checkpoint_loading : bool, checkpoint_load_path : None, best_loss_saving : bool, best_acc_saving : bool, save_each_epoch : int, checkpoint_save_path : str, shard_checkpoint_size : None) -> None:
        setup_fabric()
        self.total_epochs = total_epochs
        self.compute_type = compute_type
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.best_acc_saving = best_acc_saving
        self.best_loss_saving = best_loss_saving
        self.shard_checkpoint_loading = shard_checkpoint_loading
        self.checkpoint_load_path = checkpoint_load_path
        self.save_each_epoch = save_each_epoch
        self.checkpoint_save_path = checkpoint_save_path
        self.shard_checkpoint_size = shard_checkpoint_size
        self.dataset_max_line_length = dataset_max_line_length
        self.start_time = datetime.datetime.now()
        self.model_manager = Model_Manager(compute_type=self.compute_type, model_type=self.model_type, tokenizer_name=self.tokenizer_name, model_name=self.model_name, lora_config=lora_config, abyss=False)
        try:
            os.remove(self.checkpoint_save_path + 'model_performance.csv')
        except:
            pass
        
        
        self.tokenizer, self.model, self.class_module_name = self.model_manager.get_tokenizer_and_model()
        
        
        
        if not compute_type == '8bit':
            if not compute_type == '4bit':
                if gradient_checkpointing:
                    self.model.gradient_checkpointing_enable()
        if not compute_type == '8bit':
            if not compute_type == '4bit':
                if gradient_checkpointing:
                    self.model.gradient_checkpointing_disable()
        
        
        global train_dataset
        
        if dataset is not None:
            train_dataset = dataset
        if dataset is None:
            train_dataset = CustomDataset(dataset_path, self.tokenizer, self.dataset_max_line_length)
        
        self.DataLoader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, shuffle=self.shuffle)
        self.FabricDataLoader = fabric.setup_dataloaders(self.DataLoader)
        self.optimizer_control = optimizer_manager.optimizer_manager(self.model, optimizer_config=optimizer_config)
        self.optimizer, self.scheduler, self.gradient_clip, self.optimizer_name = self.optimizer_control.get_optimizer()
        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)
        
        
        if self.checkpoint_load_path is not None:
            if not self.shard_checkpoint_loading:
                if not self.compute_type == '8bit':
                    if not self.compute_type == '4bit':
                        print(f'{Fore.LIGHTGREEN_EX}Loading Checkpoint at {self.checkpoint_load_path}.{Fore.RESET}')
                        
                        self.model.load_state_dict(fabric.load(self.checkpoint_load_path))
        
        if self.checkpoint_load_path is not None:
            if self.shard_checkpoint_loading:
                if not self.compute_type == '8bit':
                    if not self.compute_type == '4bit':
                        print(f'{Fore.LIGHTGREEN_EX}Loading Checkpoint at {self.checkpoint_load_path}.{Fore.RESET}')
                        
                        load_sharded_checkpoint(self.model, self.checkpoint_load_path, strict=True)
                
        if self.checkpoint_load_path is not None:
            if self.compute_type == '8bit':
                print(f'{Fore.LIGHTRED_EX}Warning: Currently use cannot Load a checkpoint to continue training on 8bit, will implement it if there is a way.{Fore.RESET}')
        
        
        if self.checkpoint_load_path is not None:
            if self.compute_type == '4bit':
                print(f'{Fore.LIGHTRED_EX}Warning: Currently use cannot Load a checkpoint to continue training on 4bit, will implement it if there is a way.{Fore.RESET}')
        
        
        
    def print_trainable_parameters(self):
         print (f"{Fore.LIGHTGREEN_EX}Currently Using [({Fore.LIGHTBLUE_EX}{fabric.device}{Fore.LIGHTGREEN_EX})] As Accelerator.")
         trainable_params = 0
         all_param = 0
         for _, param in self.model.named_parameters():
             all_param += param.numel()
             if param.requires_grad:
                 trainable_params += param.numel()
         print(f"{Fore.BLUE}████████████████████\n\n{Fore.LIGHTCYAN_EX}[Model Name]\n{Fore.LIGHTMAGENTA_EX}{self.class_module_name}\n\n{Fore.BLUE}████████████████████{Fore.RESET}\n")
         if not self.compute_type == '4bit':
            print(f"\n{Fore.LIGHTCYAN_EX}[Total Parameters]\n{Fore.LIGHTMAGENTA_EX}{all_param / 1e6:.1f}M{Fore.BLUE}\n████████████████████\n\n{Fore.LIGHTCYAN_EX}[Trainable Parameters]\n{Fore.LIGHTMAGENTA_EX}{trainable_params / 1e6:.1f}M{Fore.Blue}\n████████████████████")
         if self.compute_type == '4bit':
            print(f"\n{Fore.LIGHTCYAN_EX}[Total Parameters]\n{Fore.LIGHTMAGENTA_EX}{all_param*2 / 1e6:.1f}M{Fore.BLUE}\n████████████████████\n\n{Fore.LIGHTCYAN_EX}[Trainable Parameters]\n{Fore.LIGHTMAGENTA_EX}{trainable_params*2 / 1e6:.1f}M{Fore.Blue}\n████████████████████")
        
        
    def get_current_time(self):
        end_time = datetime.datetime.now()
        time_calculation = end_time - self.start_time
        time_calculation_minutes = time_calculation.total_seconds() / 60
        time_calculation_minutes = int(time_calculation_minutes)
        formated_start_time = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
        formated_end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
        return formated_start_time, formated_end_time, time_calculation_minutes
     
     
    
    def train(self):
        self.print_trainable_parameters()
        self.model.train()
    
        progress_bar = tqdm(colour='green', position=0, desc=f"{Fore.GREEN}{Style.BRIGHT}Current Epochs", total=self.total_epochs)
        progress_bar2 = tqdm(colour='cyan', position=1, desc=f"{Fore.LIGHTRED_EX}{Style.BRIGHT}• {Fore.LIGHTCYAN_EX}{Style.BRIGHT}Avg Loss: [Pending] {Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}⌘  {Fore.LIGHTRED_EX}{Style.BRIGHT} • {Fore.LIGHTCYAN_EX}{Style.BRIGHT}Avg Accuracy: [Pending]", total=len(self.FabricDataLoader))
        
        self.best_loss = float("inf")
        self.best_accuracy = float("inf")
        self.current_train_epoch = 0
        self.current_optim_step = 0
        for i in range(self.total_epochs):
            self.epoch_accuracy = []
            self.epoch_loss = []
            for batch_idx, batch in enumerate(self.FabricDataLoader):
                input_ids, attention_mask, labels = batch
                forward = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = forward[0]
                loss.backward()
                if not self.optimizer_name == 'Adafactor':
                    clip_grad_value_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.current_optim_step += 1
                batch_acc = torch.exp(loss.detach())
                self.epoch_accuracy.append(batch_acc)
                self.epoch_loss.append(loss.detach())
                progress_bar2.update()
        
            self.perplexity = [batch_acc.cpu().item() if not np.isinf(batch_acc.cpu().item()) else 1e+8 for batch_acc in self.epoch_accuracy]
            self.perplexity = np.mean(self.perplexity)
            self.average_loss = np.mean([loss.cpu().item() for loss in self.epoch_loss])
            
            progress_bar2.set_description(f"{Fore.LIGHTRED_EX}{Style.BRIGHT} • {Fore.LIGHTCYAN_EX}{Style.BRIGHT}Avg Loss: {self.average_loss:.4f} {Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}⌘  {Fore.LIGHTRED_EX}{Style.BRIGHT} • {Fore.LIGHTCYAN_EX}{Style.BRIGHT}Avg Accuracy: {self.perplexity:.4f}")
            
            progress_bar.update()
            progress_bar2.reset()
            self.current_train_epoch += 1
            start_time, end_time, total_time = self.get_current_time()
            
            self.model_saver = Model_State_Preserver(self.model, start_time=start_time, end_time=end_time, total_time=total_time, current_train_epoch=self.current_train_epoch, save_each_epoch=self.save_each_epoch, best_loss_saving=self.best_loss_saving, best_acc_saving=self.best_acc_saving, checkpoint_save_path=self.checkpoint_save_path, shard_checkpoint_size=self.shard_checkpoint_size)
            
            get_learning_rate = self.optimizer.param_groups[0]['lr']
            get_data_sample_length = len(self.FabricDataLoader)
            
            self.model_cards = TextGenModelCard(start_time=start_time, end_time=end_time, total_time=total_time, compute_type=self.compute_type, model_type=self.model_type, model_name=self.model_name, tokenizer_name=self.tokenizer_name, optimizer_name=self.optimizer_name, learning_rate=get_learning_rate, batch_size=self.batch_size, dataset_line_max_length=self.dataset_max_line_length, total_data_samples=get_data_sample_length, total_epoch=self.total_epochs, current_epoch=self.current_train_epoch, current_step=self.current_optim_step, best_loss=self.average_loss, best_acc=self.perplexity, checkpoint_save_path=self.checkpoint_save_path)
            
            Void_Logger.log_into_csv(self.checkpoint_save_path, self.average_loss, self.perplexity, self.current_optim_step, self.current_train_epoch, total_time)
            
            
            
            self.model_cards.textgencard()
            
            if self.average_loss < self.best_loss:
                self.best_loss = self.average_loss
                self.model_saver.epoch_saving_BLS()
            
            if self.perplexity < self.best_accuracy:
                self.best_accuracy = self.perplexity
                self.model_saver.epoch_saving_BAS()
                
            if self.current_train_epoch % self.save_each_epoch == 0:
                self.model_saver.epoch_saving()
            
            
            
            
            