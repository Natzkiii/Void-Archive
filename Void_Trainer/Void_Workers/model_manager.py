import sys

sys.path.append(".")
sys.path.append("..")
import json
import torch
from colorama import Fore
#import transformers


class Model_Manager:
    def __init__(
        self, 
        compute_type: str,
        model_type : str,
        tokenizer_name: str,
        model_name: str,
        lora_config: None,
        abyss : bool,
    ) -> None:
        self.compute_type = compute_type
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.lora_config = lora_config
        self.abyss = abyss
        
    
    
    
    def config_reader(self, location : None):
        with open(location, 'r', encoding='utf-8') as file:
            file = file.read()
            json_object = json.loads(file)
            bias = json_object['bias']
            lora_alpha = json_object['lora_alpha']
            lora_dropout = json_object['lora_dropout']
            lora_r = json_object['lora_r']
            target_modules = json_object['target_modules']
            task_type = json_object['task_type']
            return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
    
    
    
    def verify_lora_config_8bit(self):
        
        verify_matches = ['Lora_8bit_S_Llama-OPT-GPTNeo', 'Lora_8bit_M_Llama-OPT-GPTNeo', 'Lora_8bit_L_Llama-OPT-GPTNeo', 'Lora_8bit_XL_Llama-OPT-GPTNeo', 'Lora_8bit_S_GPTNeoX', 'Lora_8bit_M_GPTNeoX', 'Lora_8bit_L_GPTNeoX', 'Lora_8bit_XL_GPTNeoX']
        
        if self.lora_config is not None:
            if self.lora_config == 'Lora_8bit_S_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_S_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            
            if self.lora_config == 'Lora_8bit_M_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_M_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_8bit_L_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_L_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_8bit_XL_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_XL_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_8bit_S_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_S_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_8bit_M_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_M_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_8bit_L_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_L_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_8bit_XL_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_8bit_XL_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
            
            if not self.lora_config == 'Lora_8bit_S_Llama-OPT-GPTNeo':
                if not self.lora_config == 'Lora_8bit_M_Llama-OPT-GPTNeo':
                    if not self.lora_config == 'Lora_8bit_L_Llama-OPT-GPTNeo':
                        if not self.lora_config == 'Lora_8bit_XL_Llama-OPT-GPTNeo':
                            if not self.lora_config == 'Lora_8bit_S_GPTNeoX':
                                if not self.lora_config == 'Lora_8bit_M_GPTNeoX':
                                    if not self.lora_config == 'Lora_8bit_L_GPTNeoX':
                                        if not self.lora_config == 'Lora_8bit_XL_GPTNeoX':
                                            location = self.lora_config
                                            self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=location)
                   
                    
                    
                    
    def verify_lora_config_4bit(self):
        
        verify_matches = ['Lora_4bit_S_Llama-OPT-GPTNeo', 'Lora_4bit_M_Llama-OPT-GPTNeo', 'Lora_4bit_L_Llama-OPT-GPTNeo', 'Lora_4bit_XL_Llama-OPT-GPTNeo', 'Lora_4bit_S_GPTNeoX', 'Lora_4bit_M_GPTNeoX', 'Lora_4bit_L_GPTNeoX', 'Lora_4bit_XL_GPTNeoX']
        
        if self.lora_config is not None:
            if self.lora_config == 'Lora_4bit_S_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_S_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            
            if self.lora_config == 'Lora_4bit_M_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_M_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_4bit_L_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_L_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_4bit_XL_Llama-OPT-GPTNeo':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_XL_Llama-OPT-GPTNeo.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_4bit_S_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_S_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_4bit_M_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_M_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_4bit_L_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_L_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
                
            if self.lora_config == 'Lora_4bit_XL_GPTNeoX':
                static_location = 'Void_Workers/Auto_LoraConfig/Lora_4bit_XL_GPTNeoX.json'
                self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=static_location)
            
            if not self.lora_config == 'Lora_4bit_S_Llama-OPT-GPTNeo':
                if not self.lora_config == 'Lora_4bit_M_Llama-OPT-GPTNeo':
                    if not self.lora_config == 'Lora_4bit_L_Llama-OPT-GPTNeo':
                        if not self.lora_config == 'Lora_4bit_XL_Llama-OPT-GPTNeo':
                            if not self.lora_config == 'Lora_4bit_S_GPTNeoX':
                                if not self.lora_config == 'Lora_4bit_M_GPTNeoX':
                                    if not self.lora_config == 'Lora_4bit_L_GPTNeoX':
                                        if not self.lora_config == 'Lora_4bit_XL_GPTNeoX':
                                            location = self.lora_config
                                            self.bias, self.lora_alpha, self.lora_dropout, self.lora_r, self.target_modules, self.task_type = self.config_reader(location=location)
            
    
        
            
                
                
    
    def Load_GPTNeoModel(self):
        from transformers import GPTNeoForCausalLM, GPT2TokenizerFast, BitsAndBytesConfig
        
        if self.compute_type == 'float32':
            print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
            tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
            class_module_name = 'GPTNeoForCausalLM [float32]'
        
            return tokenizer, model, class_module_name
        
        if self.compute_type == 'float16':
            print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
            tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
            model.half()
            class_module_name = 'GPTNeoForCausalLM [float16]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == '8bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16,
                )
            self.verify_lora_config_8bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'GPTNeoForCausalLM [8bit]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == '4bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
                )
            self.verify_lora_config_4bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'GPTNeoForCausalLM [4bit]'
            
            return tokenizer, model, class_module_name
        
        
    
    
    def Load_GPTNeoXModel(self):
        from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, BitsAndBytesConfig
        
        if self.compute_type == 'float32':
            print(f"{Fore.LIGHTRED_EX}Note: Loading GPTNeoX model in float32 cost lot of memory please consider using float16 for GPTNeoX since that is what the model was trained on.")
            
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoXForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
            class_module_name = 'GPTNeoXForCausalLM [float32]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == 'float16':
            
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoXForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
            model.half()
            class_module_name = 'GPTNeoXForCausalLM [float16]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == '8bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16,
                )
            
            self.verify_lora_config_8bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoXForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'GPTNeoXForCausalLM [8bit]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == '4bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
                )
            
            self.verify_lora_config_4bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(self.tokenizer_name)
            model = GPTNeoXForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'GPTNeoXForCausalLM [4bit]'
            
            return tokenizer, model, class_module_name
        
    
    
    def Load_LlamaModel(self):
        import transformers
        from transformers import LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig
        
        if not transformers.__version__ <= '4.30.0':
            print(f"{Fore.LIGHTRED_EX}Fatal Error: Llama need transformers version 4.30.0 or above but current version is {transformers.__version__}\n\nModel Manager: Exiting with error code (LLAMTVIC){Fore.RESET}")
            sys.exit(1)
        
        
        if self.compute_type == 'float32':
            print(f"{Fore.LIGHTRED_EX}Note: Loading Llama model in float32 cost lot of memory please consider using float16 for Llama since that is what the model was trained on.{Fore.RESET}")
            
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = LlamaForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
            class_module_name = 'LlamaForCausalLM [float32]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == 'float16':
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = LlamaForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
            model.half()
            class_module_name = 'LlamaForCausalLM [float16]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == '8bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16,
                )
            
            self.verify_lora_config_8bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = LlamaForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'LlamaForCausalLM [8bit]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == '4bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
                )
            
            self.verify_lora_config_4bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = LlamaForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'LlamaForCausalLM [4bit]'
                
            return tokenizer, model, class_module_name 
    
    
    def Load_OPTModel(self):
        from transformers import OPTForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        if self.compute_type == 'float32':
            print(f"{Fore.LIGHTRED_EX}Note: Loading OPT model in float32 cost lot of memory please consider using float16 for OPT since that is what the model was trained on.{Fore.RESET}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            model = OPTForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
            class_module_name = 'OPTForCausalLM [float32]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == 'float16':
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            model = OPTForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
            model.half()
            class_module_name = 'OPTForCausalLM [float16]'
            
            return tokenizer, model, class_module_name
        
        
        if self.compute_type == '8bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16,
                )
            
            self.verify_lora_config_8bit()
            
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            model = OPTForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'OPTForCausalLM [8bit]'
            
            return tokenizer, model, class_module_name
        
        
        if self.compute_type == '4bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
                )
            
            self.verify_lora_config_4bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            model = OPTForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'OPTForCausalLM [4bit]'
            
            return tokenizer, model, class_module_name
        
        
    def Load_MistralModel(self):
        import transformers
        from transformers import MistralForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig
        
        if not transformers.__version__ <= '4.34.0':
            print(f"{Fore.LIGHTRED_EX}Fatal Error: Mistral need transformers version 4.34.0 or above but current version is {transformers.__version__}\n\nModel Manager: Exiting with error code (LLAMTVIC){Fore.RESET}")
            sys.exit(1)
            
        
        if self.compute_type == 'float32':
            print(f"{Fore.LIGHTRED_EX}Note: Loading OPT model in float32 cost lot of memory please consider using float16 for OPT since that is what the model was trained on.{Fore.RESET}")
            
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = MistralForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
            class_module_name = 'MistralForCausalLM [float32]'
            
            return tokenizer, model, class_module_name
        
        if self.compute_type == 'float16':
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = MistralForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
            model.half()
            class_module_name = 'MistralForCausalLM [float16]'
            
            return tokenizer, model, class_module_name
        
        
        if self.compute_type == '8bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16,
                )
            
            self.verify_lora_config_8bit()
            
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = MistralForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'MistralForCausalLM [8bit]'
            
            return tokenizer, model, class_module_name
        
        
        if self.compute_type == '4bit':
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
                )
            
            self.verify_lora_config_4bit()
            
            config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_alpha, 
                target_modules=self.target_modules, 
                lora_dropout=self.lora_dropout, 
                bias=self.bias, 
                task_type=self.task_type
                )
            
            tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            model = MistralForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
            if not self.abyss:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            class_module_name = 'MistralForCausalLM [4bit]'
            
            return tokenizer, model, class_module_name
        
    
    
    
    def get_tokenizer_and_model(self):
        if self.model_type == 'GPTNeo':
            output = self.Load_GPTNeoModel()
            return output
        
        if self.model_type == 'GPTNeoX':
            output = self.Load_GPTNeoXModel()
            
            return output
        
        
        if self.model_type == 'Llama':
            output = self.Load_LlamaModel()
            
            return output
            
        if self.model_type == 'OPT':
            output = self.Load_OPTModel()
            
            return output
            
        if self.model_type == 'Mistral':
            output = self.Load_MistralModel()
            
            return output