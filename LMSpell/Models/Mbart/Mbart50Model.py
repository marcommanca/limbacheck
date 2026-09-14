from ModelEnum import ModelEnum
from Models.Mbart.MbartAbstractModel import MbartAbstractModel

class Mbart50Model(MbartAbstractModel):
    
    def __init__(self):
        self.model_label = ModelEnum.MBART50
        super().__init__()
