import sys
sys.path.append(".")
sys.path.append("..")
import os
import json
import torch
from colorama import Fore, Style
import transformers




def json_config_reader_8bit(lora_config : None):
    if lora_config is not None:
        if lora_config == 'Lora_8bit_Small_Llama-OPT-GPTNeo':
            static_location = 'Auto_LoraConfig/Lora_8bit_Small_Llama-OPT-GPTNeo.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        if lora_config == 'Lora_8bit_Mid_Llama-OPT-GPTNeo':
            static_location = 'Auto_LoraConfig/Lora_8bit_Mid_Llama-OPT-GPTNeo.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        if lora_config == 'Lora_8bit_Large_Llama-OPT-GPTNeo':
            static_location = 'Auto_LoraConfig/Lora_8bit_Large_Llama-OPT-GPTNeo.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        if lora_config == 'Lora_8bit_Small_GPTNeoX':
            static_location = 'Auto_LoraConfig/Lora_8bit_Small_GPTNeoX.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
            
        if lora_config == 'Lora_8bit_Mid_GPTNeoX':
            static_location = 'Auto_LoraConfig/Lora_8bit_Mid_GPTNeoX.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        if lora_config == 'Lora_8bit_Large_GPTNeoX':
            static_location = 'Auto_LoraConfig/Lora_8bit_Large_GPTNeoX.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
            
        if not lora_config == 'Lora_8bit_Small_Llama-OPT-GPTNeo':
            if not lora_config == 'Lora_8bit_Mid_Llama-OPT-GPTNeo':
                if not lora_config == 'Lora_8bit_Large_Llama-OPT-GPTNeo':
                    if not lora_config == 'Lora_8bit_Small_GPTNeoX':
                        if not lora_config == 'Lora_8bit_Mid_GPTNeoX':
                            if not lora_config == 'Lora_8bit_Large_GPTNeoX':
                                with open(lora_config, 'r', encoding='utf-8') as LC:
                                    json_lc = LC.read()
                                    config_object = json.loads(json_lc)
                                    bias = config_object['bias']
                                    lora_alpha = config_object['lora_alpha']
                                    lora_dropout = config_object['lora_dropout']
                                    lora_r = config_object['lora_r']
                                    target_modules = config_object['target_modules']
                                    task_type = config_object['task_type']
                                    return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
                
                
def json_config_reader_4bit(lora_config : None):
    if lora_config is not None:
        if lora_config == 'Lora_4bit_Small_Llama-OPT-GPTNeo':
            static_location = 'Auto_LoraConfig/Lora_4bit_Small_Llama-OPT-GPTNeo.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        if lora_config == 'Lora_4bit_Mid_Llama-OPT-GPTNeo':
            static_location = 'Auto_LoraConfig/Lora_4bit_Mid_Llama-OPT-GPTNeo.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        
        if lora_config == 'Lora_4bit_Large_Llama-OPT-GPTNeo':
            static_location = 'Auto_LoraConfig/Lora_4bit_Large_Llama-OPT-GPTNeo.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        
        if lora_config == 'Lora_4bit_Small_GPTNeoX':
            static_location = 'Auto_LoraConfig/Lora_4bit_Small_GPTNeoX.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        if lora_config == 'Lora_4bit_Mid_GPTNeoX':
            static_location = 'Auto_LoraConfig/Lora_4bit_Mid_GPTNeoX.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
        
        if lora_config == 'Lora_4bit_Large_GPTNeoX':
            static_location = 'Auto_LoraConfig/Lora_4bit_Large_GPTNeoX.json'
            with open(static_location, 'r', encoding='utf-8') as LC:
                json_lc = LC.read()
                config_object = json.loads(json_lc)
                bias = config_object['bias']
                lora_alpha = config_object['lora_alpha']
                lora_dropout = config_object['lora_dropout']
                lora_r = config_object['lora_r']
                target_modules = config_object['target_modules']
                task_type = config_object['task_type']
                return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type
            
            
        if not lora_config == 'Lora_4bit_Small_Llama-OPT-GPTNeo':
            if not lora_config == 'Lora_4bit_Mid_Llama-OPT-GPTNeo':
                if not lora_config == 'Lora_4bit_Large_Llama-OPT-GPTNeo':
                    if not lora_config == 'Lora_4bit_Small_GPTNeoX':
                        if not lora_config == 'Lora_4bit_Mid_GPTNeoX':
                            if not lora_config == 'Lora_4bit_Large_GPTNeoX':
                                with open(lora_config, 'r', encoding='utf-8') as LC:
                                    json_lc = LC.read()
                                    config_object = json.loads(json_lc)
                                    bias = config_object['bias']
                                    lora_alpha = config_object['lora_alpha']
                                    lora_dropout = config_object['lora_dropout']
                                    lora_r = config_object['lora_r']
                                    target_modules = config_object['target_modules']
                                    task_type = config_object['task_type']
                                    return bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type    
        
            





def Load_GPTNeoModel(compute_type : str, tokenizer_name : str, model_name : str, lora_config : None):
    
    from transformers import GPTNeoForCausalLM, GPT2TokenizerFast, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    if compute_type == 'float32':
        print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
        
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
        class_module_name = 'GPTNeoForCausalLM [float32]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == 'float16':
        print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
        
        
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
        model.half()
        class_module_name = 'GPTNeoForCausalLM [float16]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == '8bit':
        print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
            )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_8bit(lora_config=lora_config)
        
        
        
        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )

        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'GPTNeoForCausalLM [8bit]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == '4bit':
        print(f"{Fore.LIGHTRED_EX}Warning: We have notice GPTNeo model performing poorly in past few day's, if the loss rate is not going down try using a another type of model.{Fore.RESET}")
        
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
        )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_4bit(lora_config=lora_config)
        
        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )
        
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'GPTNeoForCausalLM [4bit]'
        
        return tokenizer, model, class_module_name
       
    
    

def Load_GPTNeoXModel(compute_type : str, tokenizer_name : str, model_name : str, lora_config : None):
    
    from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    
    if compute_type == 'float32':
        print(f"{Fore.LIGHTRED_EX}Note: Loading GPTNeoX model in float32 cost lot of memory please consider using float16 for GPTNeoX since that is what the model was trained on.")
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoXForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
        class_module_name = 'GPTNeoXForCausalLM [float32]'
        
        return tokenizer, model, class_module_name
       
    if compute_type == 'float16':
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoXForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
        model.half()
        class_module_name = 'GPTNeoXForCausalLM [float16]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
            )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_8bit(lora_config=lora_config)
        
        
        
        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )

        tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoXForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'GPTNeoXForCausalLM [8bit]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == '4bit':
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
        )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_4bit(lora_config=lora_config)
        
        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )
        
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name)
        model = GPTNeoXForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'GPTNeoXForCausalLM [4bit]'
        
        return tokenizer, model, class_module_name





def Load_LLamaModel(compute_type : str, tokenizer_name : str, model_name : str, lora_config : None):
    if not transformers.__version__ <= '4.30.0':
        print(f"{Fore.LIGHTRED_EX}Fatal Error: Llama need transformers version 4.30.0 or above but current version is {transformers.__version__}\n\nModel Manager: Exiting with error code (LLAMTVIC){Fore.RESET}")
        sys.exit(1)
    
    from transformers import LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    
    if compute_type == 'float32':
        print(f"{Fore.LIGHTRED_EX}Note: Loading Llama model in float32 cost lot of memory please consider using float16 for Llama since that is what the model was trained on.{Fore.RESET}")
        
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
        model = LlamaForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
        class_module_name = 'LlamaForCausalLM [float32]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == 'float16':
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
        model = LlamaForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
        model.half()
        class_module_name = 'LlamaForCausalLM [float16]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
            )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_8bit(Lora_Config=Lora_Config)
        
        
        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )

        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
        model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'LlamaForCausalLM [8bit]'
        
        return tokenizer, model, class_module_name
    
    
    if compute_type == '4bit':
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
        )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_4bit(Lora_Config=Lora_Config)

        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )

        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
        model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'LlamaForCausalLM [4bit]'
        
        return tokenizer, model, class_module_name 
    
    
    
    
def Load_OPTModel(compute_type : str, tokenizer_name : str, model_name : str, Lora_Config : None):
    from transformers import OPTForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    if compute_type == 'float32':
        print(f"{Fore.LIGHTRED_EX}Note: Loading OPT model in float32 cost lot of memory please consider using float16 for OPT since that is what the model was trained on.{Fore.RESET}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = OPTForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
        class_module_name = 'OPTForCausalLM [float32]'
        
        
        return tokenizer, model, class_module_name
       
    if compute_type == 'float16':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = OPTForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
        model.half()
        class_module_name = 'OPTForCausalLM [float16]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
            )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_8bit(Lora_Config=Lora_Config)
        
        
        
        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = OPTForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'OPTForCausalLM [8bit]'
        
        return tokenizer, model, class_module_name
    
    if compute_type == '4bit':
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
        )
        
        bias, lora_alpha, lora_dropout, lora_r, target_modules, task_type  = json_config_reader_4bit(Lora_Config=Lora_Config)
        
        config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias=bias, 
            task_type=task_type
            )
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = OPTForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        class_module_name = 'OPTForCausalLM [4bit]'
        
        return tokenizer, model, class_module_name