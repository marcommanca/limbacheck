from ModelEnum import ModelEnum
from Models.Xlmr.XlmrAbstractModel import XlmrAbstractModel

class XlmrLargeModel(XlmrAbstractModel):
    
    def __init__(self):
        self.model_label = ModelEnum.XLMR_LARGE
        super().__init__()
