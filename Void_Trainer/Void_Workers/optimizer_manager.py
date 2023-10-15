import torch
from torch.optim import AdamW
from transformers.optimization import Adafactor
from torch.optim.lr_scheduler import MultiStepLR
from colorama import Fore
import sys
import json

class optimizer_manager():
    def __init__(self, model, optimizer_config : None) -> None:
        self.model = model
        self.optimizer_config = optimizer_config
    
    def config_reader(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            file = file.read()
            json_loader = json.loads(file)
            optimizer_name = json_loader['optimizer_name']
            try:
                learning_rate_scheduler = json_loader['learning_rate_scheduler']
            except:
                learning_rate_scheduler = None
            try:    
                milestone = json_loader['milestone']
            except:
                if learning_rate_scheduler == 'Multistep':
                    print(f'{Fore.LIGHTRED_EX}Optimizer Manager: milestone is missing from the config file.{Fore.RESET}')
                    sys.exit(0)
                else:
                    milestone = None
            try:
                gamma = json_loader['gamma']
            except:
                if learning_rate_scheduler == 'Multistep':
                    print(f'{Fore.LIGHTRED_EX}Optimizer Manager: gamma is missing from the config file.{Fore.RESET}')
                    sys.exit(0)
                else:
                    gamma = None
            try:
                learning_rate = json_loader['learning_rate']
            except:
                print(f'{Fore.LIGHTRED_EX}Optimizer Manager: learning rate is missing from the config file.{Fore.RESET}')
                sys.exit(0)
            try:
                weight_decay = json_loader['weight_decay']
            except:
                print(f'{Fore.LIGHTRED_EX}Optimizer Manager: weight decay is missing from the config file.{Fore.RESET}')
                sys.exit(0)
            try:
                decay_rate = json_loader['decay_rate']
            except:
                if optimizer_name == 'Adafactor':
                    print(f'{Fore.LIGHTRED_EX}Optimizer Manager: decay_rate is missing from the config file.{Fore.RESET}')
                    sys.exit(0)
                else:
                    decay_rate = None
            try:
                gradient_clip = json_loader['gradient_clip']
            except:
                print(f'{Fore.LIGHTRED_EX}Optimizer Manager: gradient_clip is missing from the config file.{Fore.RESET}')
                sys.exit(0)
            try:
                scale_parameter = json_loader['scale_parameter']
                if scale_parameter == 'false':
                    scale_parameter = False
                        
                if scale_parameter == 'true':
                    scale_parameter = True
            except:
                if optimizer_name == 'Adafactor':
                    print(f'{Fore.LIGHTRED_EX}Optimizer Manager: scale_parameter is missing from the config file.{Fore.RESET}')
                    sys.exit(0)
                else:
                    scale_parameter = None
            
            try:        
                relative_step = json_loader['relative_step']
                if relative_step == 'false':
                    relative_step = False
                    
                if relative_step == 'true':
                    relative_step = True
            except:
                if optimizer_name == 'Adafactor':
                    print(f'{Fore.LIGHTRED_EX}Optimizer Manager: relative_step is missing from the config file.{Fore.RESET}')
                    sys.exit(0)
                else:
                    relative_step = None
                    
            try:
                warmup_init = json_loader['warmup_init']
                if warmup_init == 'false':
                    warmup_init = False
                    
                if warmup_init == 'true':
                    warmup_init = True
            except:
                if optimizer_name == 'Adafactor':
                    print(f'{Fore.LIGHTRED_EX}Optimizer Manager: warump_init is missing from the config file.{Fore.RESET}')
                    sys.exit(0)
                else:
                    warmup_init = None
            
            
            return optimizer_name, learning_rate_scheduler, milestone, gamma, learning_rate, weight_decay, decay_rate, gradient_clip, scale_parameter, relative_step, warmup_init
    
    def Optimizer_config_valid(self):
        
        if self.optimizer_config is not None:
            if self.optimizer_config == 'Adafactor_S_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/Adafactor_S_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            
            if self.optimizer_config == 'Adafactor_M_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/Adafactor_M_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'Adafactor_L_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/Adafactor_L_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'Adafactor_XL_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/Adafactor_XL_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'Adafactor_RS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/Adafactor_RS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'AdamW_S_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW_S_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            
            if self.optimizer_config == 'AdamW_M_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW_M_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'AdamW_L_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW_L_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'AdamW_XL_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW_XL_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'AdamW_MS_1':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW_MS_1.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            
            if self.optimizer_config == 'AdamW_MS_2':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW_MS_2.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'Adafactor_MS_1':
                self.static_path = 'Void_Workers/Auto_Optim_Config/Adafactor_MS_1.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
                
            
            if self.optimizer_config == 'Adafactor_MS_2':
                self.static_path = 'Void_Workers/Auto_Optim_Config/Adafactor_MS_2.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
            
                    
            if self.optimizer_config == 'AdamW8bit_S_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW8bit_S_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
            if self.optimizer_config == 'AdamW8bit_M_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW8bit_M_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
            if self.optimizer_config == 'AdamW8bit_L_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW8bit_L_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                
            if self.optimizer_config == 'AdamW8bit_XL_NS':
                self.static_path = 'Void_Workers/Auto_Optim_Config/AdamW8bit_XL_NS.json'
                self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.static_path)
                   
            if not self.optimizer_config == 'Adafactor_S_NS':
                if not self.optimizer_config == 'Adafactor_M_NS':
                    if not self.optimizer_config == 'Adafactor_L_NS':
                        if not self.optimizer_config == 'Adafactor_XL_NS':
                            if not self.optimizer_config == 'Adafactor_MS_1':
                                if not self.optimizer_config == 'Adafactor_MS_2':
                                    if not self.optimizer_config == 'AdamW_S_NS':
                                        if not self.optimizer_config == 'AdamW_M_NS':
                                            if not self.optimizer_config == 'AdamW_L_NS':
                                                if not self.optimizer_config == 'AdamW_XL_NS':
                                                    if not self.optimizer_config == 'AdamW_MS_1':
                                                        if not self.optimizer_config == 'AdamW_MS_2':
                                                            if not self.optimizer_config == 'AdamW8bit_S_NS':
                                                                if not self.optimizer_config == 'AdamW8bit_M_NS':
                                                                    if not self.optimizer_config == 'AdamW8bit_L_NS':
                                                                        if not self.optimizer_config == 'AdamW8bit_XL_NS':
                                                                            self.optimizer_name, self.learning_rate_scheduler, self.milestone, self.gamma, self.learning_rate, self.weight_decay, self.decay_rate, self.gradient_clip, self.scale_parameter, self.relative_step, self.warmup_init = self.config_reader(path=self.optimizer_config)
                                                                            
                
                    
        
        
    
    def AdamW_optimizer(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=self.weight_decay)
        gradient_clip = self.gradient_clip
        optimizer_name = self.optimizer_name
        
        if self.learning_rate_scheduler == 'Multistep':
            scheduler = MultiStepLR(optimizer=optimizer, milestones=self.milestone, gamma=self.gamma)
        
        if self.learning_rate_scheduler is None:
            scheduler = None 
        
        
        return optimizer, scheduler, gradient_clip, optimizer_name
    
    def Adafactor_optimizer(self):
        optimizer = Adafactor(self.model.parameters(), lr=self.learning_rate, clip_threshold=self.gradient_clip, decay_rate=self.decay_rate, weight_decay=self.weight_decay, scale_parameter=self.scale_parameter, relative_step=self.relative_step, warmup_init=self.warmup_init)
        gradient_clip = self.gradient_clip
        optimizer_name = self.optimizer_name
        
        
        if self.learning_rate_scheduler == 'Multistep':
            scheduler = MultiStepLR(optimizer=optimizer, milestones=self.milestone, gamma=self.gamma)
            
        if self.learning_rate_scheduler is None:
            scheduler = None        
        
        return optimizer, scheduler, gradient_clip, optimizer_name
    
    
    def AdamW8bit_optimizer(self):
        from bitsandbytes.optim.adam import Adam8bit
        optimizer = Adam8bit(self.model.parameters(), lr=self.learning_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=self.weight_decay, amsgrad=False, min_8bit_size=4096, percentile_clipping=100, block_wise=True)
        gradient_clip = self.gradient_clip
        optimizer_name = self.optimizer_name
        
        if self.learning_rate_scheduler == 'Multistep':
            scheduler = MultiStepLR(optimizer=optimizer, milestones=self.milestone, gamma=self.gamma)
            
        if self.learning_rate_scheduler is None:
            scheduler = None        
        
        return optimizer, scheduler, gradient_clip, optimizer_name
    
    
    
    
    
    def get_optimizer(self):
        self.Optimizer_config_valid()
        if self.optimizer_name == 'Adafactor':
            output = self.Adafactor_optimizer()
            return output
        
        if self.optimizer_name == 'AdamW':
            output = self.AdamW_optimizer()
            return output
        
        if self.optimizer_name == 'AdamW8bit':
            output = self.AdamW8bit_optimizer()
            return output
            
            