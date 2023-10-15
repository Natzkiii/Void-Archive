from Void_Workers.model_manager import Model_Manager
import lightning as Light
import torch
from colorama import Fore
from transformers.modeling_utils import load_sharded_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def setup_fabric():
    global fabric
    fabric = Light.Fabric(accelerator='auto', devices='auto', strategy='auto', precision='32')
    fabric.launch()



class Void_Inference:
    def __init__(self, inference_type : str, inference_model_path : str, shard_checkpoint_loading : bool, compute_type : str, f16_to_8bit : bool, f16_to_4bit, backup_mode : bool, model_type : str, tokenizer_name : str, model_name : str, lora_config : None, prompts : bool, system_prompt : str, character_greetings : str, log_dict : str, max_new_tokens : int, min_new_tokens : int, do_sample : bool, num_beams : int, top_p : int, temperature : int, repetition_penalty : int, use_cache : bool, early_stopping : bool):
        if not backup_mode:
            setup_fabric()
        self.inference_type = inference_type
        self.inference_model_path = inference_model_path
        self.shard_checkpoint_loading = shard_checkpoint_loading
        self.compute_type = compute_type
        self.f16_to_8bit = f16_to_8bit
        self.f16_to_4bit = f16_to_4bit
        self.backup_mode = backup_mode
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.lora_config = lora_config
        self.prompts = prompts
        self.system_prompt = system_prompt
        self.character_greeting = character_greetings
        self.log_dict = log_dict
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.use_cache = use_cache
        self.early_stopping = early_stopping
        self.backup_device = torch.device('cuda')
        
        if not self.f16_to_8bit:
            if not self.f16_to_4bit:
                if not self.compute_type == '8bit':
                    if not self.compute_type == '4bit':
                        self.model_manager = Model_Manager(self.compute_type, self.model_type, self.tokenizer_name, self.model_name, self.lora_config, abyss=True)
                        self.tokenizer, self.model, self.class_module_name = self.model_manager.get_tokenizer_and_model()
        
        if self.f16_to_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        if self.f16_to_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            
            
        if self.compute_type == '8bit':
            print(f'{Fore.LIGHTRED_EX}Warning: Currently You can use this to inference with only float 32 and float 16 models. will implement the ablity to inference with lora models in a future update.{Fore.RESET}')
            sys.exit(0)
            
        if self.compute_type == '4bit':
            print(f'{Fore.LIGHTRED_EX}Warning: Currently You can use this to inference with only float 32 and float 16 models. will implement the ablity to inference with lora models in a future update.{Fore.RESET}')
            sys.exit(0)
        
        if self.backup_device:
            self.model = self.model
        else:
            self.model = fabric.setup(self.model)
        
        
        if self.inference_model_path is not None:
            if not self.shard_checkpoint_loading:
                self.model.load_state_dict(fabric.load(self.inference_model_path))
        
        if self.inference_model_path is not None:
            if self.shard_checkpoint_loading:
                load_sharded_checkpoint(self.model, self.inference_model_path, strict=True)
        
        
        
    def TextGeneration_prompt(self):
        
        self.model.eval()
        
        
        omit_logs = ['']
        
        system = ['<|system|>']
        system.append(self.system_prompt)
        character_greeting = ['<|model|>']
        character_greeting.append(self.character_greeting)
        system = ''.join(system)
        character_init_prompt = ''.join(character_greeting)
        splited_greeting = character_init_prompt.split('<|model|>')
        finalized_greet = ''.join(splited_greeting)
        print(f'{Fore.LIGHTCYAN_EX}Greeting: {finalized_greet}')
        
        
        with torch.no_grad():
            while True:
                user_input = input(f'{Fore.LIGHTGREEN_EX}User: ')
                finalized_prompt = system + character_init_prompt + '\n<|user|>' + user_input + '\n<|model|>'
                omit_logs.append('\\n<|user|>' + user_input)
                
               
                input_ids = self.tokenizer.encode(finalized_prompt, return_tensors='pt').to(fabric.device)
                attention_mask = torch.ones_like(input_ids).to(fabric.device)
        
                generate = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=self.early_stopping
                    )
                    
                output = self.tokenizer.decode(generate[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                output = output.split('<|model|>')[-1]
                output = output.split('<|user|>')[0]
                output = output.split('<|user|>')[0]
                output = output.replace('\\n', '\n')
                log_ready_out = output.replace('\n', '\\n')
                    
                omit_logs.append('\\n<|model|>' + log_ready_out)
                    
                print(f'{Fore.LIGHTCYAN_EX}Model: {output}{Fore.RESET}')
                if user_input == '[(=save=)]':
                        print(f'{Fore.LIGHTRED_EX}Warning: This will replace the old TextGeneration_Omit.txt with the new one, backup the old one if you want, i will implement a auto system later.')
                        finalized_omit_logs = ''.join(omit_logs)
                        with open(self.log_dict + 'TextGeneration_Omit.txt', 'w', encoding='utf-8') as file:
                            file = file.write(finalized_omit_logs)
                    
                
                    
    
    
    
    def TextGeneration_no_prompt(self):
        
        self.model.eval()
        
        
        omit_logs = ['']
        
        
        with torch.no_grad():
            while True:
                user_input = input(f'{Fore.LIGHTGREEN_EX}User: {Fore.RESET}')
                
                omit_logs.append('\\n<|user|>' + user_input)
                
               
                input_ids = self.tokenizer.encode(user_input, return_tensors='pt').to(fabric.device)
                attention_mask = torch.ones_like(input_ids).to(fabric.device)
        
                generate = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=self.early_stopping
                    )
                    
                output = self.tokenizer.decode(generate[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                output = output.split('<|model|>')[-1]
                output = output.split('<|user|>')[0]
                output = output.split('<|user|>')[0]
                output = output.replace('\\n', '\n')
                log_ready_out = output.replace('\n', '\\n')
                    
                omit_logs.append('\\n<|model|>' + log_ready_out)
                    
                print(f'{Fore.LIGHTCYAN_EX}Model: {output}{Fore.RESET}')
                if user_input == '[(=save=)]':
                        print(f'{Fore.LIGHTRED_EX}Warning: This will replace the old TextGeneration_Omit.txt with the new one, backup the old one if you want, i will implement a auto system later.')
                        finalized_omit_logs = ''.join(omit_logs)
                        with open(self.log_dict + 'TextGeneration_Omit.txt', 'w', encoding='utf-8') as file:
                            file = file.write(finalized_omit_logs)
            
          
          
     
    def TextGeneration_prompt_backup(self):
        
        self.model.eval()
        
        
        omit_logs = ['']
        
        system = ['<|system|>']
        system.append(self.system_prompt)
        character_greeting = ['<|model|>']
        character_greeting.append(self.character_greeting)
        system = ''.join(system)
        character_init_prompt = ''.join(character_greeting)
        splited_greeting = character_init_prompt.split('<|model|>')
        finalized_greet = ''.join(splited_greeting)
        print(f'{Fore.LIGHTCYAN_EX}Greeting: {finalized_greet}')
        
        
        with torch.no_grad():
            while True:
                user_input = input(f'{Fore.LIGHTGREEN_EX}User: ')
                finalized_prompt = system + character_init_prompt + '\n<|user|>' + user_input + '\n<|model|>'
                omit_logs.append('\\n<|user|>' + user_input)
                
               
                input_ids = self.tokenizer.encode(finalized_prompt, return_tensors='pt').to(self.backup_device)
                attention_mask = torch.ones_like(input_ids).to(self.backup_device)
        
                generate = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=self.early_stopping
                    )
                    
                output = self.tokenizer.decode(generate[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                output = output.split('<|model|>')[-1]
                output = output.split('<|user|>')[0]
                output = output.split('<|user|>')[0]
                output = output.replace('\\n', '\n')
                log_ready_out = output.replace('\n', '\\n')
                    
                omit_logs.append('\\n<|model|>' + log_ready_out)
                    
                print(f'{Fore.LIGHTCYAN_EX}Model: {output}{Fore.RESET}')
                if user_input == '[(=save=)]':
                        print(f'{Fore.LIGHTRED_EX}Warning: This will replace the old TextGeneration_Omit.txt with the new one, backup the old one if you want, i will implement a auto system later.')
                        finalized_omit_logs = ''.join(omit_logs)
                        with open(self.log_dict + 'TextGeneration_Omit.txt', 'w', encoding='utf-8') as file:
                            file = file.write(finalized_omit_logs)
                    
                
                    
    
    
    
    def TextGeneration_no_prompt_backup(self):
        
        self.model.eval()
        
        
        omit_logs = ['']
        
        
        with torch.no_grad():
            while True:
                user_input = input(f'{Fore.LIGHTGREEN_EX}User: {Fore.RESET}')
                
                omit_logs.append('\\n<|user|>' + user_input)
                
               
                input_ids = self.tokenizer.encode(user_input, return_tensors='pt').to(self.backup_device)
                attention_mask = torch.ones_like(input_ids).to(self.backup_device)
        
                generate = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=self.early_stopping
                    )
                    
                output = self.tokenizer.decode(generate[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                output = output.split('<|model|>')[-1]
                output = output.split('<|user|>')[0]
                output = output.split('<|user|>')[0]
                output = output.replace('\\n', '\n')
                log_ready_out = output.replace('\n', '\\n')
                    
                omit_logs.append('\\n<|model|>' + log_ready_out)
                    
                print(f'{Fore.LIGHTCYAN_EX}Model: {output}{Fore.RESET}')
                if user_input == '[(=save=)]':
                        print(f'{Fore.LIGHTRED_EX}Warning: This will replace the old TextGeneration_Omit.txt with the new one, backup the old one if you want, i will implement a auto system later.')
                        finalized_omit_logs = ''.join(omit_logs)
                        with open(self.log_dict + 'TextGeneration_Omit.txt', 'w', encoding='utf-8') as file:
                            file = file.write(finalized_omit_logs)
          
          
          
          
          
          
          
          
          
          
          
    
          
    def TextGeneration_chat_prompt(self):
        
        self.model.eval()
        
        
        chat_logs = ['']
        
        
        system = ['<|system|>']
        system.append(self.system_prompt)
        character_greeting = ['<|model|>']
        character_greeting.append(self.character_greeting)
        system = ''.join(system)
        character_init_prompt = ''.join(character_greeting)
        splited_greeting = character_init_prompt.split('<|model|>')
        finalized_greet = ''.join(splited_greeting)
        print(f'{Fore.LIGHTCYAN_EX}Greeting: {finalized_greet}')
            
        finalize_chat_logs = ''.join(chat_logs)
        
        with torch.no_grad():
            while True:
                user_input = input(f'{Fore.LIGHTGREEN_EX}User: ')
                finalized_prompt = system + character_init_prompt + finalize_chat_logs + '\n<|user|>' + user_input + '\n<|model|>'
                
                chat_logs.append('\\n<|user|>' + user_input)
                
               
                input_ids = self.tokenizer.encode(finalized_prompt, return_tensors='pt').to(fabric.device)
                attention_mask = torch.ones_like(input_ids).to(fabric.device)
                
                    
                generate = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=self.early_stopping
                    )
                    
                output = self.tokenizer.decode(generate[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                output = output.split('<|model|>')[-1]
                output = output.split('<|user|>')[0]
                output = output.split('<|user|>')[0]
                output = output.replace('\\n', '\n')
                log_ready_out = output.replace('\n', '\\n')
                    
                chat_logs.append('\\n<|model|>' + log_ready_out)
                    
                print(f'{Fore.LIGHTCYAN_EX}Model: {output}{Fore.RESET}')
                    
                if user_input == '[(=save=)]':
                    print(f'{Fore.LIGHTRED_EX}Warning: This will replace the old TextGeneration_Omit.txt with the new one, backup the old one if you want, i will implement a auto system later.')
                        
                    with open(self.log_dict + 'TextGeneration_Omit.txt', 'w', encoding='utf-8') as file:
                        file = file.write(finalize_chat_logs)      
          
          
          
          
          
          
          
    
          
          
          
          
    
    def TextGeneration_chat_prompt_backup(self):
        
        self.model.eval()
        
        
        chat_logs = ['']
        
        
        system = ['<|system|>']
        system.append(self.system_prompt)
        character_greeting = ['<|model|>']
        character_greeting.append(self.character_greeting)
        system = ''.join(system)
        character_init_prompt = ''.join(character_greeting)
        splited_greeting = character_init_prompt.split('<|model|>')
        finalized_greet = ''.join(splited_greeting)
        print(f'{Fore.LIGHTCYAN_EX}Greeting: {finalized_greet}')
            
        finalize_chat_logs = ''.join(chat_logs)
        
        with torch.no_grad():
            while True:
                user_input = input(f'{Fore.LIGHTGREEN_EX}User: ')
                finalized_prompt = system + character_init_prompt + finalize_chat_logs + '\n<|user|>' + user_input + '\n<|model|>'
                
                chat_logs.append('\\n<|user|>' + user_input)
                
               
                input_ids = self.tokenizer.encode(finalized_prompt, return_tensors='pt').to(self.backup_device)
                attention_mask = torch.ones_like(input_ids).to(self.backup_device)
                
                    
                generate = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=self.early_stopping
                    )
                    
                output = self.tokenizer.decode(generate[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                output = output.split('<|model|>')[-1]
                output = output.split('<|user|>')[0]
                output = output.split('<|user|>')[0]
                output = output.replace('\\n', '\n')
                log_ready_out = output.replace('\n', '\\n')
                    
                chat_logs.append('\\n<|model|>' + log_ready_out)
                    
                print(f'{Fore.LIGHTCYAN_EX}Model: {output}{Fore.RESET}')
                    
                if user_input == '[(=save=)]':
                    print(f'{Fore.LIGHTRED_EX}Warning: This will replace the old TextGeneration_Omit.txt with the new one, backup the old one if you want, i will implement a auto system later.')
                        
                    with open(self.log_dict + 'TextGeneration_Omit.txt', 'w', encoding='utf-8') as file:
                        file = file.write(finalize_chat_logs)
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
                            
    def inference(self):
        if self.inference_type == 'TxtGen':
            if self.prompts:
                if not self.backup_mode:
                    self.TextGeneration_prompt()
        
        if self.inference_type == 'TxtGen':
            if not self.prompts:
                if not self.backup_mode:
                    self.TextGeneration_no_prompt()
                    
        
        if self.inference_type == 'TxtGen':
            if self.prompts:
                if self.backup_mode:
                    self.TextGeneration_prompt_backup()
                    
        if self.inference_type == 'TxtGen':
            if not self.prompts:
                if self.backup_mode:
                    self.TextGeneration_no_prompt_backup()
                
                
        if self.inference_type == 'Chat':
            if self.prompts:
                if not self.backup_mode:
                    self.TextGeneration_chat_prompt()
                    
                    
        if self.inference_type == 'Chat':
            if self.prompts:
                if self.backup_mode:
                    self.TextGeneration_chat_prompt_backup()
                
        