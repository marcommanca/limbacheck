import pytest
import torch
import pandas as pd
from unittest.mock import MagicMock, patch
from Models.Mt5.Mt5LargeModel import Mt5LargeModel

@pytest.fixture
def mock_model():
    with patch("Models.Mt5.Mt5AbstractModel.MT5ForConditionalGeneration.from_pretrained") as mock_model_loader, \
         patch("Models.Mt5.Mt5AbstractModel.T5TokenizerFast.from_pretrained") as mock_tokenizer_loader:

        mock_model_loader.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 123
        mock_tokenizer.all_special_ids = [0, 1, 2]
        mock_tokenizer.decode.side_effect = lambda ids, skip_special_tokens: f"decoded_{ids.tolist()}"
        mock_tokenizer_loader.return_value = mock_tokenizer

        model = Mt5LargeModel()
        model.device = "cpu"
        return model

@patch("Models.EncoderDecoderAbstract.tqdm")
def test_predict_single_input(mock_tqdm_module, mock_model):

    mock_model.decode = MagicMock()
    mock_model.correct(max_length=10, batch_size=1, shuffle=False, input_set="input example")
    mock_model.decode.assert_called_once()

@patch("Models.EncoderDecoderAbstract.tqdm")
def test_predict_multiple_inputs(mock_tqdm_module, mock_model):

    mock_model.decode = MagicMock()
    mock_model.correct(max_length=10, batch_size=2, shuffle=False, input_set=["Text A", "Text B"])
    mock_model.decode.assert_called_once()

def test_decode_function(mock_model):
    originals = [torch.tensor([1, 2, 3])]
    predictions = [torch.tensor([4, 5, 6])]
    labels = [torch.tensor([4, 5, 6])]

    with patch("Utils.PrePostProcessingUtils.PrePostProcessingUtils.remove_special_tokens", side_effect=lambda tokens, all_ids, keep_ids: tokens), \
         patch("Utils.PrePostProcessingUtils.PrePostProcessingUtils.clean_zwj", side_effect=lambda x: x), \
         patch("Utils.PrePostProcessingUtils.PrePostProcessingUtils.save_dataframe") as mock_save_dataframe:

        mock_model.decode(originals, predictions, labels, "test")

        mock_save_dataframe.assert_called_once()
        df_arg = mock_save_dataframe.call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert not df_arg.empty
        assert all(col in df_arg.columns for col in ["Original", "Corrected", "Expected"])
