from ModelEnum import ModelEnum
from Models.Mt5.Mt5AbstractModel import Mt5AbstractModel

class Mt5LargeModel(Mt5AbstractModel):
    
    def __init__(self):
        self.model_label = ModelEnum.MT5_LARGE
        super().__init__()
