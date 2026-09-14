from Utils.DatasetUtils import DatasetUtils
from Data.LMSpellDataset import LMSpellDataset
from torch.utils.data import DataLoader

class LMDataLoader:

    def __init__(
            self,
            tokenizer,
            dataset = LMSpellDataset,
            train_path = None,
            val_path = None,
            test_path = None,
            dataset_size = 1,
            train_max_length = 128,
            test_max_length = 128,
            train_batch_size = 16,
            test_batch_size = 16,
        ):

        self.dataset = dataset
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
        dataset = DatasetUtils(train_path=train_path, val_path=val_path, test_path=test_path, dataset_size=dataset_size)
        
        if dataset.train_dataset is not None:
            train_dataset = self.dataset(dataset.train_dataset, tokenizer, train_max_length)
            train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True)
            self.train_dataloader = train_dataloader

        if dataset.val_dataset is not None:
            val_dataset = self.dataset(dataset.val_dataset, tokenizer, train_max_length)
            val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True)
            self.val_dataloader = val_dataloader

        if dataset.test_dataset is not None:
            test_dataset = self.dataset(dataset.test_dataset, tokenizer, test_max_length)
            test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True)
            self.test_dataloader = test_dataloader