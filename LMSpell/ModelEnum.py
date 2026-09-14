from enum import Enum
from Models.ModelAbstract import ModelAbstract
from typing import Dict, Type


class ModelEnum(Enum):
    """
    Enum class for all the models supported by the 
    Neural Spell Checker Library
    """
    # Fine-tuned models
    # These checkpoints will be released upon the acceptance of the paper

    # Llama 3.1 models
    LLAMA_31_8B = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    LLAMA_31_8B_INSTRUCT = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    
    # Llama 3.2 models
    LLAMA_32_1B = "unsloth/Llama-3.2-1B-bnb-4bit"
    LLAMA_32_1B_INSTRUCT = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    LLAMA_32_3B = "unsloth/Llama-3.2-3B-bnb-4bit"
    LLAMA_32_3B_INSTRUCT = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    
    # Gemma 2 models
    GEMMA_2_2B = "unsloth/gemma-2-2b-bnb-4bit"
    GEMMA_2_2B_INSTRUCT = "unsloth/gemma-2-2b-it-bnb-4bit"
    GEMMA_2_9B = "unsloth/gemma-2-9b-bnb-4bit"
    GEMMA_2_9B_INSTRUCT = "unsloth/gemma-2-9b-it-bnb-4bit"

    # MT5 models
    MT5_SMALL = "google/mt5-small"
    MT5_BASE = "google/mt5-base"
    MT5_LARGE = "google/mt5-large"
    MT5_XL = "google/mt5-xl"
    MT5_XXL = "google/mt5-xxl"
    
    # MBART models
    MBART50 = "facebook/mbart-large-50"
    
    # SinBERT models
    SINBERT_SMALL = "NLPC-UOM/SinBERT-small"
    SINBERT_LARGE = "NLPC-UOM/SinBERT-large"
    
    # XLMR models
    XLMR_BASE = "FacebookAI/xlm-roberta-base"
    XLMR_LARGE = "FacebookAI/xlm-roberta-large"

    def getModelClass(self) -> ModelAbstract:
        """Return the model class associated with the enum value"""

        from Models.Llama.LlamaAbstractModel import LlamaAbstractModel
        from Models.Llama.Llama318BModel import Llama318BModel

        from Models.Gemma.GemmaAbstractModel import GemmaAbstractModel
        from Models.Gemma.Gemma29BModel import Gemma29BModel

        from Models.Mt5.Mt5AbstractModel import Mt5AbstractModel
        from Models.Mt5.Mt5BaseModel import Mt5BaseModel
        from Models.Mt5.Mt5LargeModel import Mt5LargeModel

        from Models.Mbart.Mbart50Model import Mbart50Model

        from Models.SinBert.SinBertAbstractModel import SinBertAbstractModel
        from Models.Xlmr.XlmrAbstractModel import XlmrAbstractModel
        from Models.Xlmr.XlmrLargeModel import XlmrLargeModel
        # import other model classes as needed

        modelClasses: Dict[ModelEnum, ModelAbstract] = {
            # Fine-tuned models
            # ModelEnum.LLAMA31_FT_SI: LlamaAbstractModel,
            # ModelEnum.GEMMA2_FT_SI: GemmaAbstractModel,
            # ModelEnum.MT5_FT_SI: Mt5AbstractModel,
            # ModelEnum.MT5_XL_FT_SI: Mt5AbstractModel,
            # ModelEnum.MBART_FT_SI: MbartAbstractModel,
            # ModelEnum.SINBERT_FT_SI: SinBertAbstractModel,
            # ModelEnum.XLMR_FT_SI: XlmrAbstractModel,

            # Llama 3.1 models
            ModelEnum.LLAMA_31_8B: Llama318BModel,
            ModelEnum.LLAMA_31_8B_INSTRUCT: LlamaAbstractModel,

            # Llama 3.2 models
            ModelEnum.LLAMA_32_1B: LlamaAbstractModel,
            ModelEnum.LLAMA_32_1B_INSTRUCT: LlamaAbstractModel,
            ModelEnum.LLAMA_32_3B: LlamaAbstractModel,
            ModelEnum.LLAMA_32_3B_INSTRUCT: LlamaAbstractModel,

            # Gemma 2 models
            ModelEnum.GEMMA_2_2B: GemmaAbstractModel,
            ModelEnum.GEMMA_2_2B_INSTRUCT: GemmaAbstractModel,
            ModelEnum.GEMMA_2_9B: Gemma29BModel,
            ModelEnum.GEMMA_2_9B_INSTRUCT: GemmaAbstractModel,

            # mT5 models
            ModelEnum.MT5_SMALL: Mt5AbstractModel,
            ModelEnum.MT5_BASE: Mt5BaseModel,
            ModelEnum.MT5_LARGE: Mt5LargeModel,
            ModelEnum.MT5_XL: Mt5AbstractModel,
            ModelEnum.MT5_XXL: Mt5AbstractModel,

            # mBART models
            ModelEnum.MBART50: Mbart50Model,

            # SinBERT models
            ModelEnum.SINBERT_SMALL: SinBertAbstractModel,
            ModelEnum.SINBERT_LARGE: SinBertAbstractModel,
            
            # XLMR models
            ModelEnum.XLMR_BASE: XlmrAbstractModel,
            ModelEnum.XLMR_LARGE: XlmrLargeModel,

            # Add other model-class mappings here
        }
        return modelClasses.get(self)
    
    def get_model_path(self) -> str:
        """Return the model path"""
        return self.value
    
    @staticmethod
    def get_supported_models():
        """Return a list of all supported model names"""
        return [model.value for model in ModelEnum]