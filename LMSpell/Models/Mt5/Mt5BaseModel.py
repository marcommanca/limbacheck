from ModelEnum import ModelEnum
from Models.Mt5.Mt5AbstractModel import Mt5AbstractModel

class Mt5BaseModel(Mt5AbstractModel):
    
    def __init__(self):
        self.model_label = ModelEnum.MT5_BASE
        super().__init__()
