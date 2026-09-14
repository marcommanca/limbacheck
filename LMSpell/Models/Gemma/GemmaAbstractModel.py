from Models.DecoderAbstract import DecoderAbstract
from ModelEnum import ModelEnum

class GemmaAbstractModel(DecoderAbstract):
    def __init__(self):
        # TODO
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