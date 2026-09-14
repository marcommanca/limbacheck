import unittest
from ModelEnum import ModelEnum
from Models.Llama.Llama318BModel import Llama318BModel
from Models.Mt5.Mt5LargeModel import Mt5LargeModel

class ModelEnumTest(unittest.TestCase):

    def test_gets_model_class(self):
        self.assertIs(ModelEnum.getModelClass(ModelEnum.LLAMA_31_8B), Llama318BModel)
        self.assertIs(ModelEnum.getModelClass(ModelEnum.MT5_LARGE), Mt5LargeModel)

    def test_gets_model_name(self):
        self.assertEqual(ModelEnum.LLAMA_31_8B.get_model_path(), "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
        self.assertEqual(ModelEnum.GEMMA_2_9B.get_model_path(), "unsloth/gemma-2-9b-bnb-4bit")
        self.assertEqual(ModelEnum.MT5_LARGE.get_model_path(), "google/mt5-large")

    def test_gets_supported_models(self):
        self.assertIn("unsloth/Meta-Llama-3.1-8B-bnb-4bit", ModelEnum.get_supported_models())
        self.assertIn("unsloth/gemma-2-9b-bnb-4bit", ModelEnum.get_supported_models())
        self.assertIn("google/mt5-large", ModelEnum.get_supported_models())
        
if __name__ == '__main__':
    unittest.main()