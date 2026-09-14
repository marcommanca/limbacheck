class DatasetUtils:
    
    def __init__(self, train_path = None, val_path = None, test_path = None, dataset_size = 1, seed = 42, zwj_fix = False):
        from datasets import Dataset
        import pandas as pd

        self.dataset_size = dataset_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if train_path is not None:
            train = pd.read_csv(train_path).dropna()
            train = train.sample(frac=self.dataset_size, random_state=seed).reset_index(drop=True)
            if zwj_fix:
                train = self.fix_zwj(train)
            print(f"Train shape: {train.shape}")
            self.train_dataset = Dataset.from_pandas(train)
            
        if val_path is not None:
            val = pd.read_csv(val_path).dropna()
            if zwj_fix:
                val = self.fix_zwj(val)
            print(f"Val shape: {val.shape}")
            self.val_dataset = Dataset.from_pandas(val)
            
        if test_path is not None:
            test = pd.read_csv(test_path).dropna()
            if zwj_fix:
                test = self.fix_zwj(test)
            print(f"Test shape: {test.shape}")
            self.test_dataset = Dataset.from_pandas(test)
  
    @staticmethod
    def fix_zwj(df):
        from Utils.ConfigUtils import ConfigUtils

        for col in ConfigUtils.get_dataset_col_names():
            df[col] = df[col].str.replace('\u200D', '<ZWJ>')

        return df
