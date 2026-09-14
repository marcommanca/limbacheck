from unittest.mock import patch, MagicMock
from ModelEnum import ModelEnum
from Models.Mt5.Mt5LargeModel import Mt5LargeModel

@patch("Models.Mt5.Mt5AbstractModel.MT5ForConditionalGeneration.from_pretrained")
@patch("Models.Mt5.Mt5AbstractModel.T5TokenizerFast.from_pretrained")
def test_initialize_model(mock_tokenizer, mock_model):

    mock_model_instance = MagicMock(name="mT5-large-mock")
    mock_model_instance.to.return_value = mock_model_instance
    mock_model.return_value = mock_model_instance
    mock_tokenizer.return_value = MagicMock(name="mT5-large-tokenizer-mock")

    instance = Mt5LargeModel()

    assert instance.model_label is ModelEnum.MT5_LARGE
    assert instance.model is mock_model.return_value
    assert instance.tokenizer is mock_tokenizer.return_value