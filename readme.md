```Welcome To Inferno Hex Project```

**Description**

```Inferno Hex Is "Minimalist" Code To Train And Inference With GPT-2 Language Model```

**Features**

```
1: Support CSV And TSV File Formats

2: Can Run 1B Parameter GPT Model Using This Code In Single 16GB GPU

3: Have A Chathistory In Chat.py

4: Low Memory Usage

5: Faster Loading Times

6: Can Be Used In CPU Or GPU But Would Not Recommend Training Using CPU
```

**Requirements**

```
Torch: 2.0+
Transformers: 4.29+
Tqdm: 4.64+
Colorama: 0.4.5+
```

**How To Run**

```
1: Create A Folder Named "config" And Put GPT-2 Config File Inside It

2: Use Vocabgen.py To Generate A Vocab.json And Merges.txt Then Create A Folder Named "tokenizer" And Place Them Inside That

3: Go And Edit Config.json File's Vocab Size To The Size Of The Vocab You Create Using Vocabgen.py [Important] You Can Edit Vocabgen.py To Change The Vocab Size And Etc

4: Change Settings To Your Desire [Important] Increasing Batch Size Too High Might Cause CUDA Out Of Memory Error's

5: Run Train.py

6: If You Need To Load A Checkpoint You Can Change 'model_path': 'saved_models/distgpt2_0.pt' In Settings

7: For Inference Run chat.py

8: For Inference You Can Load The Model Using 'inference_model_path': 'saved_models/distgpt2_1.pt' In Settings

9: If You Need To Change Where The Model Is Being Saved You Can Change 'saved_model_path': 'saved_models/distgpt2_1.pt' In Settings
```
