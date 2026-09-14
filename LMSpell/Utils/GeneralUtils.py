class GeneralUtils:

    @staticmethod
    def clean_memory():
        import gc
        import torch

        gc.collect() # Collect garbage
        torch.cuda.empty_cache() # Empty the CUDA cache

    @staticmethod
    def delete_folder(folder_path):
        import shutil
        import os
        
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Removed folder: {folder_path}")