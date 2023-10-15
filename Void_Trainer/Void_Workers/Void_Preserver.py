from colorama import Fore


class Model_State_Preserver:
    def __init__(
        self,
        model,
        start_time,
        end_time,
        total_time,
        current_train_epoch : int,
        save_each_epoch : int,
        best_loss_saving : bool,
        best_acc_saving : bool,
        checkpoint_save_path : None,
        shard_checkpoint_size: None,
        
    ) -> None:
        self.model = model
        self.start_time = start_time
        self.end_time = end_time
        self.total_time = total_time
        self.current_train_epoch = current_train_epoch
        self.best_loss_saving = best_loss_saving
        self.best_acc_saving = best_acc_saving
        self.checkpoint_save_path = checkpoint_save_path
        self.shard_checkpoint_size = shard_checkpoint_size
        self.save_each_epoch = save_each_epoch
    
    
    def epoch_saving_BLS(self):
        # Best loss saving with No shard checkpoints, No Merge And Save.
        if self.current_train_epoch % self.save_each_epoch == 0:
            if self.best_loss_saving:
                if self.shard_checkpoint_size is None:
                    self.model.save_pretrained(self.checkpoint_save_path)
                    print(f"\n\n{Fore.LIGHTBLUE_EX}███████████████████████\n\n{Fore.LIGHTGREEN_EX}[Model Saving Complete]{Fore.LIGHTCYAN_EX}\n\nStart Date and Time: {self.start_time}\nEnd Date and Time: {self.end_time}\nTotal Minutes: {self.total_time}{Fore.RESET}\n")
                                
        # Best loss_saving with shard checkpoints and No Merge And Save.
        if self.current_train_epoch % self.save_each_epoch == 0:
            if self.best_loss_saving:
                if self.shard_checkpoint_size is not None:
                    self.model.save_pretrained(self.checkpoint_save_path, max_shard_size=self.shard_checkpoint_size)
                    print(f"\n\n{Fore.LIGHTBLUE_EX}███████████████████████\n\n{Fore.LIGHTGREEN_EX}[Model Saving Complete]{Fore.LIGHTCYAN_EX}\n\nStart Date and Time: {self.start_time}\nEnd Date and Time: {self.end_time}\nTotal Minutes: {self.total_time}{Fore.RESET}\n")
        
                                                  
                                                  
    def epoch_saving_BAS(self):
        # Best acc saving with No shard checkpoints, No Merge And Save.
        if self.current_train_epoch % self.save_each_epoch == 0:
            if self.best_acc_saving:
                if self.shard_checkpoint_size is None:
                    self.model.save_pretrained(self.checkpoint_save_path)
                    print(f"\n\n{Fore.LIGHTBLUE_EX}███████████████████████\n\n{Fore.LIGHTGREEN_EX}[Model Saving Complete]{Fore.LIGHTCYAN_EX}\n\nStart Date and Time: {self.start_time}\nEnd Date and Time: {self.end_time}\nTotal Minutes: {self.total_time}{Fore.RESET}\n")
                                
        # Best acc saving with shard checkpoints and No Merge And Save.
        if self.current_train_epoch % self.save_each_epoch == 0:
            if self.best_acc_saving:
                    if self.shard_checkpoint_size is not None:
                        self.model.save_pretrained(self.checkpoint_save_path, max_shard_size=self.shard_checkpoint_size)
                        print(f"\n\n{Fore.LIGHTBLUE_EX}███████████████████████\n\n{Fore.LIGHTGREEN_EX}[Model Saving Complete]{Fore.LIGHTCYAN_EX}\n\nStart Date and Time: {self.start_time}\nEnd Date and Time: {self.end_time}\nTotal Minutes: {self.total_time}{Fore.RESET}\n")
        

    
    def epoch_saving(self):
        # epoch saving with No shard checkpoints, No Merge And Save.
        if self.current_train_epoch % self.save_each_epoch == 0:
            if not self.best_acc_saving:
                if not self.best_loss_saving:
                    if self.shard_checkpoint_size is None:
                        self.model.save_pretrained(self.checkpoint_save_path)
                        print(f"\n\n{Fore.LIGHTBLUE_EX}███████████████████████\n\n{Fore.LIGHTGREEN_EX}[Model Saving Complete]{Fore.LIGHTCYAN_EX}\n\nStart Date and Time: {self.start_time}\nEnd Date and Time: {self.end_time}\nTotal Minutes: {self.total_time}{Fore.RESET}\n")
                                
        # epoch saving with shard checkpoints and No Merge And Save.
        if self.current_train_epoch % self.save_each_epoch == 0:
            if not self.best_acc_saving:
                if not self.best_loss_saving:
                    if self.shard_checkpoint_size is not None:
                        self.model.save_pretrained(self.checkpoint_save_path, max_shard_size=self.shard_checkpoint_size)
                        print(f"\n\n{Fore.LIGHTBLUE_EX}███████████████████████\n\n{Fore.LIGHTGREEN_EX}[Model Saving Complete]{Fore.LIGHTCYAN_EX}\n\nStart Date and Time: {self.start_time}\nEnd Date and Time: {self.end_time}\nTotal Minutes: {self.total_time}{Fore.RESET}\n")
        