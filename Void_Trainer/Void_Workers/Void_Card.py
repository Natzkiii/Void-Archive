class TextGenModelCard:
    def __init__(self,
                 start_time,
                 end_time,
                 total_time,
                 compute_type : str,
                 model_type : str,
                 model_name : str,
                 tokenizer_name : str,
                 optimizer_name : str,
                 learning_rate,
                 batch_size : int,
                 dataset_line_max_length : int,
                 total_data_samples : int,
                 total_epoch : int,
                 current_epoch : int,
                 current_step : int,
                 best_loss,
                 best_acc,
                 checkpoint_save_path
                 
        ) -> None:
        
        self.start_time = start_time
        self.end_time = end_time
        self.total_time = total_time
        self.compute_type = compute_type
        self.model_type = model_type
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.dataset_max_length = dataset_line_max_length
        self.batch_size = batch_size
        self.total_data_samples = total_data_samples
        self.total_epoch = total_epoch
        self.current_epoch = current_epoch
        self.current_step = current_step
        self.best_loss = best_loss
        self.best_acc = best_acc
        self.checkpoint_path = checkpoint_save_path + 'Model_Card.md'
        
        
    def textgencard(self):
        card = f'### This is auto generated by void archive library at the end of each epoch.\n\nModel Compute Type: {self.compute_type}\nModel Type: {self.model_type}\nPre Trained Model Used In Fine Tuning: {self.model_name}\nPre Trained Tokenizer Used In Fine Tuning: {self.tokenizer_name}\nTrained Batch Size: {self.batch_size}\nTotal Data Samples Used: {self.total_data_samples * self.batch_size}\nMax Fine Tuned Length: {self.dataset_max_length} Tokens\nOptimizer Used In Training: {self.optimizer_name}\nLearning Rate Used In Training : {self.learning_rate}\nTotal Expected Epochs: {self.total_epoch}\nCurrent Epoch: {self.current_epoch}\nTotal Optimizer Step: {self.current_step}\nBest Average Accuracy: {str(self.best_acc)[:5]}\nBest Average Loss: {str(self.best_loss)[:5]}\nStarted Training At: {self.start_time}\nEnded Training At: {self.end_time}\nTotal Time Taken In Minutes: {self.total_time}'
        
        with open(self.checkpoint_path, 'w', encoding='utf-8') as file:
            file = file.write(card)
            
    
        
        
