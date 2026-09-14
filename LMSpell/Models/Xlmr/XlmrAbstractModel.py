from Models.EncoderAbstract import EncoderAbstract
from transformers import AutoTokenizer, AutoConfig, XLMRobertaForMaskedLM
from accelerate import Accelerator

class XlmrAbstractModel(EncoderAbstract):
    def __init__(self):
        super().__init__()
        model_path = self.model_label.get_model_path()
        config = AutoConfig.from_pretrained(model_path)
        self.model = XLMRobertaForMaskedLM.from_pretrained(model_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

        self.seperator_token = "</s>"
        self.special_tokens_to_add=['<ZWJ>']

        if self.special_tokens_to_add is not None:
            self.tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens_to_add})

        self.accelerator = Accelerator()
