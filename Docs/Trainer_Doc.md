## ⌘ Trainer Fully Explained ⌘

### ⚙️ Compute Type.

- *Compute type here refere to the model's torch.dtype.*

- *You can currently train any model's we support in* *`[float32, float16, 8bit, 4bit.]`*

- *Simply change the compute_type to what you need from the above list.*

- *Using float32 will cost almost double the memory of using float16.*


### ⚙️ Model Type.

- *Model Type here refere to the model's architecture.*

- *You can currently train these architecture base models.* *`['GPTNeo', 'GPTNeoX', 'Llama', 'OPT', 'Mistral']`*

- *Simply change the model_type to the model you need to use.*

### ⚙️ Tokenizer Name And Model Name.

- *Tokenizer Name And Model Name here refere to the hugging face pre trained models and tokenizer's*

- *Simply change the tokenizer_name and model_name to your desired model you want to train.*

### ⚙️ Optimizer Config.

- *Optimizer Config here refere to the optimizer that is going to be used in training the model.*

- *We currently support three optimizer's.*

- *Supported optimizers are* *`['AdamW', 'Adafactor', 'AdamW8bit']`*

- *Here is the list of predifined config's:*

  ***AdamW Configs***

  *`['AdamW_S_NS', 'AdamW_M_NS', 'AdamW_L_NS', 'AdamW_XL_NS', 'AdamW_MS_1', 'AdamW_MS_2']`*

  ***Adafactor Configs***

  *`['Adafactor_S_NS', 'Adafactor_M_NS', 'Adafactor_L_NS', 'Adafactor_XL_NS', 'Adafactor_MS_1', 'Adafactor_MS_2', 'Adafactor_RS']`*

  ***AdamW8bit Configs***

  *`['AdamW8bit_S_NS', 'AdamW8bit_M_NS', 'AdamW8bit_L_NS', 'AdamW8bit_XL_NS']`*

- *To use a pre defined config just change the optimizer_config to one of these above.*

- *Remember to use AdamW8bit when training a 8bit or 4bit model.*

- *You can give a custom json file as the optimizer config if you want.* ***Note: For A Examples Of How The Json FIles Should Be Formated. [Click Here](https://github.com/VINUK0/Void-Archive/tree/Void-Trainer/Void_Trainer/Void_Workers/Auto_Optim_Config)***

### ⚙️ Lora Config.

- *Lora Config here refere to the config of the peft model.*

- *If you are not training the model in 8bit or 4bit please keep it `None`.*

- *Lora aka Peft training in 8bit and 4bit currently support all 5 model's that are available.*

- *There is many pre difined lora config's for you to use.*

- *Try using larger config's for smaller model's for better results.*

- *Here is the list of pre defined config's that are available.*

  ***Config's for Llama/OPT/GPTNeo 8bit***

  *`['Lora_8bit_S_Llama-OPT-GPTNeo', 'Lora_8bit_M_Llama-OPT-GPTNeo', 'Lora_8bit_L_Llama-OPT-GPTNeo', 'Lora_8bit_XL_Llama-OPT-GPTNeo']`*

  ***Config's for GPTNeoX 8bit***

  *`['Lora_8bit_S_GPTNeoX', 'Lora_8bit_M_GPTNeoX', 'Lora_8bit_L_GPTNeoX', 'Lora_8bit_XL_GPTNeoX']`*


  ***Config's for Llama/OPT/GPTNeo 4bit***

  *`['Lora_4bit_S_Llama-OPT-GPTNeo', 'Lora_4bit_M_Llama-OPT-GPTNeo', 'Lora_4bit_L_Llama-OPT-GPTNeo', 'Lora_4bit_XL_Llama-OPT-GPTNeo']`*

  ***Config's for GPTNeoX 4bit***

  *`['Lora_4bit_S_GPTNeoX', 'Lora_4bit_M_GPTNeoX', 'Lora_4bit_L_GPTNeoX', 'Lora_4bit_XL_GPTNeoX']`*

- *To use a pre defined config just change the lora_config to one of these above.*

- *You can give a custom json file as the lora config if you want.* ***Note: For A Examples Of How The Json FIles Should Be Formated. [Click Here](https://github.com/VINUK0/Void-Archive/tree/Void-Trainer/Void_Trainer/Void_Workers/Auto_LoraConfig)***


### ⚙️ Dataset, Dataset_Path And Dataset_max_line_length

- *Dataset here refere to the custom_dataset class that you create to make your dataset usable with pytorch.*

- *If you leave the dataset `None` in the trainer it will default to our custom_dataset handler that support `CSV, TSV, DB` file types.*

- *If you are going to use a custom_dataset of your own just pass your custom_dataset class as the dataset without wrapping it in torch dataloader*

- *You can check our custom_dataset for example. [Click Here](https://github.com/VINUK0/Void-Archive/blob/Void-Trainer/Void_Trainer/Void_Workers/Void_Dispatcher.py)*

- *we currently only support using one file as the dataset so the dataset path should look like this `\notebooks\examples\train\data.db`* *You can use a custom file as the dataset it don't need to be a data.db.*

- *Dataset_max_line_length is the pre trained tokenizer's setting that decide the longest length of each line.*

- *Dataset_max_line_length is counted in tokens, `1024 tokens = 4096 characters` this is a estimate and may not be accurate.*

- *As you increase the dataset_max_line_length it will also increase the memory usage.*


### ⚙️ Batch Size, Num Wokers And Shuffle

- *Batch Size here refere to the size that your dataset's total samples will be divided.*

- *Increasing the batch_size will result in faster training but also increase the memory usage.*

- *Num Workers here refere to the pytorch dataloader's workers.*

- *Adding a large amount of workers will result in unexpected outcomes.*

- *If you enable shuffle your data in the dataset will be shuffled.*

- *To enable shuffle set shuffle to `True`.*


### ⚙️ Gradient Checkpointing

- *Wether to use gradient checkpoint or not.*

- *Default to False.*

- *Using gradient checkpointing will result in slower training but will reduce the memory usage.*

- *To enable gradient checkpoint set gradient checkpoint to `True`*

### ⚙️ Shard Checkpoint Loading

- *Use this only if you use a custom shard_checkpoint_size when you saved a checkpoint.*

- *If you enable this make sure to set the checkpoint_load_path to the folder where the shared checkpoints are located and not to a single file.*

- *To enable set it to `True`*


### ⚙️ Checkpoint Load Path

- *If you don't have a checkpoint to load set checkpoint load path to `None`*

- *When you have shard checkpoint loading `False` you need to provide the path to your checkpoint file. like shown down below*
  *`notebooks/examples/train/saved_models/pytorch_model.bin`*

- *Checkpoint Loading is disabled when using 8bit and 4bit training.*


### ⚙️ Best Loss Saving And Best Acc Saving

- *If best loss saving or best acc saving is set to `True` the model will only be saved if the loss or accuracy is lower than before.*

### ⚙️ Checkpoint Save Path

- *The location where the trained model should be saved.*



### ⚙️ Shard Checkpoint Size

- *If this is set to `shard_checkpoint_size='100MB'` it will split the model into smaller part's, you can also use `shard_checkpont_size='1GB'`*