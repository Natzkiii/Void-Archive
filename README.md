## Welcome to [Void Archive V0.0.1] library.

### ⌘ Description ⌘
​
- *[Void Archive] library aim to make training of a machine learning model easy as possible so anyone could train a model with or without python and machine learning knowledge.

- *Currently only VINUK01 and psw01 are the only people working on the library.*

- *Any helps to improve the library will be greatly appreciated.*

- *Note: Please do not open issues in github unless there is a actual error.*

### ⌘ Features ⌘

- [✅] ***Support For Training*** ***`Float32, Float16, NF8bit, NF4bit`******.***

- [✅] ***Currently Supported Models Are*** ***`GPTNeo, GPTNeoX, OPT, Llama, Mistral`.***

- [✅] ***Support For Inference*** ***`Float32, Float16`******.***

- [❌] ***Support For Inference with LoraModel's Are Coming Soon.***

- [✅] ***Sharded Checkpoint Loading And Saving.***

- [✅] ***Pre Defined Config's For Training Lora Models With Ability To Use Your Own Custom Config.***

- [✅] ***Pre Defined Config's For Optimizer's For Better Training With Ability To Use Your Own Custom Config.***

- [✅] ***Currently Supported Optimizers Are*** ***`AdamW, Adafactor, AdamW8bit`.***

- [✅] ***Built In Custom Dataset Handler That Support's*** ***`CSV, TSV, DB`.***

- [✅] ***Model Performance Logging At Each Epoch.***

- [✅] ***Auto Model Card Generation At Each Epoch With Necessary Information.***

- [✅] ***TXT File To DB Converter In Case You Don't Have Dataset Or Don't Know How To.***

- [❌] ***Dataset File Formatter/Converter Might Be Available In The Future.***


### ⌘ How To Run ⌘

*1. Install it by cloning the github or hopefully pip in the future (i am busy a litte).*

*2. for training a model create a python file in your desired name and mirror the code down below.*

***Note: For A Full Explanation Click here.[Trainer_Doc](https://github.com/VINUK0/Void-Archive/blob/Void-Trainer/Docs/Trainer_Doc.md)***

```python
from Void_Trainer.Void_Trainer import Void_Trainer


Trainer = Void_Trainer(compute_type='float32',
                       model_type='OPT',
                       tokenizer_name='facebook/opt-350m',
                       model_name='facebook/opt-350m', 
                       optimizer_config='Adafactor_M_NS',
                       lora_config=None,
                       dataset=None,
                       dataset_path='/notebooks/examples/data.db',
                       dataset_max_line_length=512,
                       batch_size=2,
                       num_workers=2,
                       shuffle=False,
                       total_epochs=20,
                       gradient_checkpointing=False,
                       shard_checkpoint_loading=False,
                       checkpoint_load_path=None,
                       best_loss_saving=False,
                       best_acc_saving=False,
                       save_each_epoch=5,
                       checkpoint_save_path='/notebooks/examples/saved_models/',
                       shard_checkpoint_size=None)

Trainer.train()

```

*3. for inference/talking with the trained models, create a python file in your desired name and mirror the code down below.*

***Note: For A Full Explanation Click here.[Inference_Doc](https://www.markdownguide.org)***

```python

from Void_Trainer.Void_Inference import Void_Inference

Inference = Void_Inference(inference_type='Chat',
                           inference_model_path='/notebooks/examples/saved_models/pytorch_model.bin',
                           shard_checkpoint_loading=False,
                           compute_type='float32',
                           f16_to_8bit=False,
                           f16_to_4bit=False,
                           backup_mode=False,
                           model_type='OPT',
                           tokenizer_name='facebook/opt-350m',
                           model_name='facebook/opt-350m',
                           lora_config=None,
                           prompts=True,
                           system_prompt='',
                           character_greetings='',
                           log_dict='/notebooks/examples/saved_models/',
                           max_new_tokens=512,
                           min_new_tokens=50,
                           do_sample=True,
                           num_beams=1,
                           top_p=0.95,
                           temperature=0.1,
                           repetition_penalty=1.0,
                           use_cache=True,
                           early_stopping=False)


Inference.inference()


```