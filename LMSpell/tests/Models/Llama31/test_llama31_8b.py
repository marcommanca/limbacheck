# import pytest
# import sys
# from unittest.mock import patch, MagicMock, PropertyMock

# sys.modules["unsloth"] = MagicMock()
# sys.modules["unsloth.FastLanguageModel"] = MagicMock()

# from Models.Llama.Llama318BModel import Llama318BModel
# from Models.DecoderAbstract import DecoderAbstract
# from Models.ModelAbstract import ModelAbstract

# # Make a fixture for the Llama318BModel with mocks
# @pytest.fixture(autouse=True, scope="module")
# def llama_model():
#     with patch("unsloth.FastLanguageModel.FastLanguageModel.from_pretrained") as mock_from_pretrained, \
#          patch("unsloth.FastLanguageModel.FastLanguageModel.for_inference") as mock_for_inference:

#         mock_model = MagicMock()
#         mock_tokenizer = MagicMock()
#         mock_from_pretrained.return_value = (mock_model, mock_tokenizer)

#         type(mock_model).generation_config = PropertyMock(return_value=MagicMock(get_generation_mode=lambda: "test_mode"))
#         mock_model.generate.return_value = [MagicMock()]
#         mock_tokenizer.decode.return_value = "corrected text"

#         model = Llama318BModel()

#         yield model, mock_model, mock_tokenizer, mock_from_pretrained, mock_for_inference


# def test_correct_single_text(llama_model):
#     model, mock_model, mock_tokenizer, mock_from_pretrained, mock_for_inference = llama_model

#     result = model.correct("අයෙම සලකන්න")
#     assert isinstance(result, list)
#     assert result[0] == "corrected text"
#     mock_for_inference.assert_called_once()
#     mock_model.generate.assert_called_once()


# def test_correct_multiple_texts(llama_model):
#     model, mock_model, mock_tokenizer, mock_from_pretrained, mock_for_inference = llama_model

#     texts = ["අයෙම සලකන්න", "මට අවශ්‍යයි"]
#     result = model.correct(texts)
#     assert isinstance(result, list)
#     assert len(result) == 2
#     assert all(r == "corrected text" for r in result)


# def test_get_model_and_tokenizer(llama_model):
#     model, mock_model, mock_tokenizer, _, _ = llama_model

#     assert model.get_model() == mock_model
#     assert model.get_tokenizer() == mock_tokenizer


# def test_abstract_methods_implemented(llama_model):
#     model, _, _, _, _ = llama_model

#     # These methods are placeholders so they return None
#     assert model.correctFromFile("path") is None
#     assert model.evaluate("text", "target") is None
#     assert model.evaluateFromFile("src_file", "target_file") is None
#     assert model.load_model() is None
#     assert model.save_model() is None
#     assert model.get_model_name() is None
#     assert model.load_tokenizer() is None
#     assert model.save_tokenizer() is None


# def test_prompt_formatting(llama_model):
#     model, _, _, _, _ = llama_model

#     text = "මට අවශ්‍යයි"
#     prompt = model.PROMPT.format(text, "")
#     assert "මට අවශ්‍යයි" in prompt
#     assert "Output" in prompt
