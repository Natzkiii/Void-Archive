import pandas as pd
import os


def log_into_csv(checkpoint_save, epoch_loss, epoch_accuracy, total_step, total_epoch, total_time_taken):
    
    data = {
    'Epoch Loss': [],
    'Epoch Accuracy': [],
    'Total Step': [],
    'Total Epoch': [],
    'Total Time Taken': []
    }
    
    

    data['Epoch Loss'].append(str(epoch_loss)[:7])
    data['Epoch Accuracy'].append(str(epoch_accuracy)[:7])
    data['Total Step'].append(total_step)
    data['Total Epoch'].append(total_epoch)
    data['Total Time Taken'].append(total_time_taken)
    
    df = pd.DataFrame(data)
    df.to_csv(checkpoint_save + 'model_performance.csv', mode='a', header=not os.path.exists(checkpoint_save + 'model_performance.csv'), index=False)
    
    