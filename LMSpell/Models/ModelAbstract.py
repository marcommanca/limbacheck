import torch
import pandas as pd
from abc import ABC, abstractmethod
from Utils.ConfigUtils import ConfigUtils
from Exceptions.EvaluateModelException import EvaluateModelException

class ModelAbstract(ABC):
    @abstractmethod
    def __init__(self):
        '''
        Conctrete classes should define following attributes:
        - model_label
        - model
        - tokenizer
        - seed
        - max_seq_length
        - language
        '''
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = 'Sinhala'   # get it dynamically
    
    @abstractmethod
    def correct(self, src: str, target: str = None):
        pass

    @abstractmethod
    def correctFromFile(self, src: str, target: str = None):
        pass
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def get_model_name(self):
        pass
    
    @abstractmethod
    def load_tokenizer(self):
        pass
    
    @abstractmethod
    def save_tokenizer(self):
        pass
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model

    @staticmethod    
    def process_input(input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str = None):
        """
        Process the input set and target set. The input set should be a string, a list of strings or a DataFrame.
        The target set should be a string or a list of strings.
        """

        dataset_col_names = ConfigUtils.get_dataset_columns()

        if isinstance(input_set, pd.DataFrame):
            return input_set, True
            
        evaluate_flag = False
        data = []

        if isinstance(input_set, str):

            text = input_set
            evaluate_flag = target_set is not None and isinstance(target_set, str)
            expected = target_set if evaluate_flag else ""
            data = [{dataset_col_names[0]: text, dataset_col_names[1]: expected}]
            
        elif isinstance(input_set, list) and isinstance(input_set[0], str):

            evaluate_flag = (target_set is not None and 
                                isinstance(target_set, list) and 
                                len(target_set) == len(input_set))
            
            data = [{dataset_col_names[0]: s.strip(), dataset_col_names[1]: t.strip()} for s, t in zip(input_set, target_set)] if evaluate_flag else [{dataset_col_names[0]: s.strip(), dataset_col_names[1]: ""} for s in input_set]

        input_set = pd.DataFrame(data)
        return input_set, evaluate_flag