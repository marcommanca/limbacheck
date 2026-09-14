import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock

from Data.LMSpellDataset import LMSpellDataset

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text": ["Input Sentence 1", "Input Sentence 2"],
        "expected": ["Expected Sentence 1", "Expected Sentence 2"]
    })

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    return tokenizer

def test_len(sample_df, mock_tokenizer):
    dataset = LMSpellDataset(sample_df, mock_tokenizer)
    assert len(dataset) == 2

def test_getitem_structure(sample_df, mock_tokenizer):
    dataset = LMSpellDataset(sample_df, mock_tokenizer)
    item = dataset[0]
    
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape == item["labels"].shape


def test_get_method(sample_df, mock_tokenizer):
    dataset = LMSpellDataset(sample_df, mock_tokenizer)
    assert dataset.get() == 2