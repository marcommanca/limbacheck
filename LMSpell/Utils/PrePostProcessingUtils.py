import pandas as pd

class PrePostProcessingUtils:

    ZWJ: str = '<ZWJ>'
    ZWJ_UNICODE: str = '\u200D'

    @staticmethod
    def remove_special_tokens(tokens, all_special_ids, special_token_ids_to_keep):
        import torch

        # This was the original line. GPT suggested the change to supress the warning.
        # tokens_tensor = torch.tensor(tokens.clone().detach(), dtype=torch.int64)
        tokens_tensor = tokens.clone().detach().to(dtype=torch.int64)
        mask = (tokens_tensor == special_token_ids_to_keep) | (~torch.isin(tokens_tensor, all_special_ids))
        filtered_tokens = tokens_tensor[mask]

        return filtered_tokens.tolist()

    @staticmethod
    def clean_zwj(input_data):
        # For some reason the ZWJ token is decoded with a additional space in the end.
        # Regex matches '<ZWJ>' with or without a trailing space
        zwj_pattern = rf'{PrePostProcessingUtils.ZWJ}\s?'   # '<ZWJ>\s?'

        return input_data.replace(zwj_pattern, PrePostProcessingUtils.ZWJ_UNICODE, regex=True)
    
    @staticmethod
    def save_dataframe(df: 'pd.DataFrame', dir: str):
        print(df.head(3))
        df.to_excel(f'{dir}/test_results.xlsx', index=False)
        df.to_csv(f'{dir}/test_results.csv', index=False)