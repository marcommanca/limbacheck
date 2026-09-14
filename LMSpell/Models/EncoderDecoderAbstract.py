import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from Utils.ConfigUtils import ConfigUtils
from Utils.EvaluateUtils import EvaluateUtils
from LMSpellException import LMSpellException
from Models.ModelAbstract import ModelAbstract
from Data.LMSpellDataset import LMSpellDataset
from Utils.PrePostProcessingUtils import PrePostProcessingUtils

class EncoderDecoderAbstract(ModelAbstract):
    
    def correct(self, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str | pd.DataFrame = None, output_dir: str ="outputs",  max_length: int = 128, batch_size: int = 8, shuffle: bool = False):
        """
        Corrects the text. The input set should be a string, a list of strings or a DataFrame.
        Perform the evaluation if the target set is provided.
        """
        os.makedirs(os.path.join(os.getcwd(), output_dir), exist_ok=True)
        input_set, evaluate_flag = self.process_input(input_set, target_set)

        dataset = LMSpellDataset(input_set, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.model.eval()
        originals, predictions, labels = [], [], []

        with tqdm(dataloader, leave=True) as pbar:
            for batch in pbar:
                with torch.no_grad():
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                original = batch["input_ids"]
                prediction = torch.argmax(outputs.logits, dim=-1)
                label = batch.get("labels") #this is only present when we are testing. its not there when we are predicting better to have a mechanism to prevent this
                if batch.get("labels") is not None:
                    labels.extend(label)
                originals.extend(original), predictions.extend(prediction)

        results_df = self.decode(originals, predictions, labels, output_dir)

        if evaluate_flag:
            eval_df = results_df.iloc[:, :2].copy()
            if isinstance(target_set, pd.DataFrame):
                target_col = target_set.iloc[:, 0] if target_set.shape[1] > 0 else pd.Series([])
            elif isinstance(target_set, list):
                target_col = pd.Series(target_set)
            elif isinstance(target_set, str):
                target_col = pd.Series([target_set])

            eval_df[ConfigUtils.get_results_columns()[2]] = target_col.reset_index(drop=True)

            print("Evaluating the outputs...")
            EvaluateUtils.evaluate_from_dataframe(eval_df, output_dir)

        return results_df


    def decode(self, originals, predictions, labels, output_dir):
        """
        Decode a tensor of tensors and save it to a file.
        """

        skip_special_tokens = False

        # This is different in mbart model. Same for xlmr, sinbert and mt5. We don't need to do this in large models.
        special_token_id_to_keep = self.tokenizer.convert_tokens_to_ids(PrePostProcessingUtils.ZWJ)
        all_special_ids = torch.tensor(self.tokenizer.all_special_ids, dtype=torch.int64).to(self.device)

        special_token_tensor = torch.tensor([special_token_id_to_keep], dtype=torch.int64).to(self.device)

        originals, predictions, labels = [
            [PrePostProcessingUtils.remove_special_tokens(tokens.to(self.device), all_special_ids, special_token_tensor) for tokens in data]
            for data in (originals, predictions, labels)
        ]
        
        originals_decoded, predictions_decoded, labels_decoded = [
            [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '').strip() for ids in data]
            for data in (originals, predictions, labels)
        ]

        result_col_names = ConfigUtils.get_results_columns()

        results_data = {
            result_col_names[0]: originals_decoded,
            result_col_names[1]: predictions_decoded,
            result_col_names[2]: labels_decoded,
        }

        # Write CSV inside the main process condition
        results_df = pd.DataFrame(results_data)

        # When using BERT models, these character were present in the output. They throws errors when saving in excel format
        results_df = results_df.apply(lambda col: col.map(lambda x: x.replace('.\uffff', '').replace('\x11', '') if isinstance(x, str) else x))

        print("Replacing ZWJ token")
        for column in result_col_names:
            results_df[column] = PrePostProcessingUtils.clean_zwj(results_df[column])

        PrePostProcessingUtils.save_dataframe(results_df, output_dir)

        return results_df

    def correctFromFile(self, src: str, target: str = None, output_dir: str = "outputs", max_length: int = 128, batch_size: int = 8, shuffle: bool = False):
        import pandas as pd
        """
        Corrects the text from a file. The file should be in the format of
        a CSV or TXT file. If the file is a CSV, it should have two columns: 'text' and 'expected'.
        If the file is a TXT, it should have one line per sentence. If target is provided,
        it will be used for evaluation. The output will be saved in the specified output directory.
        """

        if src.endswith('.csv'):
            df = pd.read_csv(src)
            src = df[ConfigUtils.get_dataset_columns()[0]].tolist()
            target = df[ConfigUtils.get_dataset_columns()[1]].tolist()

        elif src.endswith('.txt'):
            with open(src, 'r', encoding='utf-8') as file:
                src = file.readlines()
            if target is not None:
                with open(target, 'r', encoding='utf-8') as file:
                    target = file.readlines()
        else:
            raise LMSpellException("Unsupported file format. Only .csv and .txt are supported.") from None
                
        return self.correct(src, target, output_dir, max_length, batch_size, shuffle)