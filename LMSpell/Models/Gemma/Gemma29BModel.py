from Models.Gemma.GemmaAbstractModel import GemmaAbstractModel
from ModelEnum import ModelEnum

class Gemma29BModel(GemmaAbstractModel):

    def __init__(self):
        self.model_label = ModelEnum.GEMMA_2_9B

        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        self.seed = 42      # get as a parameter
        self.do_sample = True       # get as a parameter

        super().__init__()
    
    def load_model(self):
        pass
    
    def save_model(self):
        pass
    
    def get_model_name(self):
        pass
    
    def load_tokenizer(self):
        pass
    
    def save_tokenizer(self):
        pass
