import numpy as np
import os
from Utils.GeneralUtils import GeneralUtils

class EarlyStopping():

    def __init__(self, epochs, patience, exp_name, best_loss=np.inf):
        self.best_loss = best_loss
        self.best_model = None
        self.total_epochs = epochs
        self.epoch = 1
        self.best_epoch = 1
        self.no_improvement = 0
        self.patience = patience
        self.history = {'train_loss': [], 'val_loss': []}
        self.exp_name = exp_name
        self.exp_dir = os.path.join(os.getcwd(), self.exp_name)

    def step(self, train_loss, val_loss, model):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            self.best_model = model
            self.no_improvement = 0
        else:
            self.no_improvement += 1
            
        self.cleanup()
        if self.no_improvement >= self.patience or self.epoch >= self.total_epochs:
            return True  # Stop training
        else:
            self.epoch += 1
            return False
    
    def cleanup(self):
        if (self.epoch - self.patience - 1) > 0:
            pre_epoch_folder = os.path.join(self.exp_dir, f'epoch_{self.epoch - self.patience - 1}')
            GeneralUtils.delete_folder(pre_epoch_folder)
        
    
    def summary(self):
        for epoch in range(1, self.epoch + 1):
            if epoch != self.best_epoch:
                epoch_folder = os.path.join(self.exp_dir, f'epoch_{epoch}')
                GeneralUtils.delete_folder(epoch_folder)
        
        summary = "Stopped at epoch {}, best epoch was {}, best model was {} ".format(self.epoch, self.best_epoch, self.best_model)
        print(summary)
        
    def get_best_model(self):
        if self.best_model is not None: 
            print(f"Latest epoch folder path found: {self.best_model}")
            return self.best_model        
        return self.find_latest_epoch_folder(f'exp_{self.exp_name}')


    @staticmethod
    def find_latest_epoch_folder(folder_name):
        import glob

        current_path = os.getcwd()
        # Find all epoch directories
        print(f"Searching for epoch directories in {folder_name} in {current_path}.....")
        exp_dir_pattern = os.path.join(current_path, folder_name, "epoch_*")
        epoch_dirs = glob.glob(exp_dir_pattern)
        
        if not epoch_dirs:
            print("No epoch directories found.")
            return None

        # Extract epoch numbers and sort numerically
        epoch_dirs.sort(key=lambda x: int(x.split("_")[-1]))
        # Get the directory with the highest epoch number
        latest_epoch_dir = epoch_dirs[-1]
        print(f"Latest epoch folder path found: {latest_epoch_dir}")
        return latest_epoch_dir
