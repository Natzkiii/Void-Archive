SETTINGS = {
    'raw_model': 'gpt2',
    'model_path': 'saved_models/gpt_1.pt', # Load The Saved Model's Checkpoint For Continues Training
    'saved_model_path': 'saved_models/gpt_2.pt', # Where The Fine Tuned Model Save
    'Dataset_path': 'data/output.tsv', # Load The Training Data
    'epochs': 10000, # Number Of Epochs
    'batch_size': 16,
    'num_workers': 4,
    'saveEvery': 1 # How Many Epoch Before A Save
}
inference = {
    'inference_model_path': 'saved_models/gpt_4.pt', # Load The Model For Inference
    'inference_device': 'cpu',
}
Optimizer_Settings = {
    'lr': 2e-5, # Learning Rate Of The Model
    'grad_clip': 0.5, # Gradient Clip Threshold
    'decay_rate': 0.8, # Decay Rate
    'beta1': None, # Part Of Adam/AdamW In Most Cases Just Don't Change 
    'weight_decay': 0.1, # Weight Decay Rate For Preventing Overfitting
    'relative_step': False, # If You Want To Use It Please Set The [lr] to None
    'scale_parameter': False, # If You Want To Use It Please Set The [lr] to None
    'warmup_init': False, # Slowly Increase The Learning Rate
}


Seed = {
   'train_seed': 42,
   'infer_seed': 42
}
