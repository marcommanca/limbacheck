import torch
import pandas as pd
import tempfile
import os

from Utils.PrePostProcessingUtils import PrePostProcessingUtils

def test_remove_special_tokens():
    tokens = torch.tensor([1, 2, 3, 99, 100, 101])
    all_special_ids = torch.tensor([99, 100, 101])
    special_token_ids_to_keep = 100

    result = PrePostProcessingUtils.remove_special_tokens(tokens, all_special_ids, special_token_ids_to_keep)

    assert isinstance(result, list)
    assert 99 not in result
    assert 101 not in result
    assert 100 in result
    assert set(result).issubset(set(tokens.tolist()))


def test_clean_zwj_removes_pattern():
    dirty_series = pd.Series([
        "this is <ZWJ> ",         # trailing space
        "multiple <ZWJ> <ZWJ> ",  # multiple occurrences
        "none here",
        "<ZWJ>at start and end<ZWJ> "
    ])
    cleaned = PrePostProcessingUtils.clean_zwj(dirty_series)

    assert isinstance(cleaned, pd.Series)
    for original, cleaned_val in zip(dirty_series, cleaned):
        assert "<ZWJ>" not in cleaned_val
        assert "<ZWJ> " not in cleaned_val
        if "ZWJ" in original:
            assert PrePostProcessingUtils.ZWJ_UNICODE in cleaned_val

def test_clean_zwj_handles_no_match():
    input_series = pd.Series(["hello", "world", "nothing to clean"])
    output = PrePostProcessingUtils.clean_zwj(input_series)
    assert input_series.equals(output)


def test_save_dataframe():
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        PrePostProcessingUtils.save_dataframe(df, tmpdir)

        excel_path = os.path.join(tmpdir, "test_results.xlsx")
        csv_path = os.path.join(tmpdir, "test_results.csv")

        assert os.path.exists(excel_path)
        assert os.path.exists(csv_path)

        df_excel = pd.read_excel(excel_path)
        df_csv = pd.read_csv(csv_path)

        pd.testing.assert_frame_equal(df.head(3), df_excel)
        pd.testing.assert_frame_equal(df.head(3), df_csv)
