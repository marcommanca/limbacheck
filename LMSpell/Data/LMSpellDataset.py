import pandas as pd
from torch.utils.data import Dataset
from Utils.ConfigUtils import ConfigUtils

class LMSpellDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        if not isinstance(dataframe, pd.DataFrame):
            dataframe = pd.DataFrame(dataframe)
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        dataset_columns = ConfigUtils.get_dataset_columns()
        inputs = self.tokenizer(item[dataset_columns[0]], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        targets = self.tokenizer(item[dataset_columns[1]], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        inputs['labels'] = targets['input_ids']
        return {key: val.squeeze() for key, val in inputs.items()}
    
    def get(self):
        return len(self.dataframe)
