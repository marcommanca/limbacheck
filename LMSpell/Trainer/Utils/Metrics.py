import os
import torch
import pandas as pd
from ModelEnum import ModelEnum
from torch.utils.data import DataLoader
from Utils.ConfigUtils import ConfigUtils
from Utils.EvaluateUtils import EvaluateUtils
from Utils.PrePostProcessingUtils import PrePostProcessingUtils

class Metrics:
    def __init__(self, data_loader: DataLoader, exp_dir: str, ZWJ_Fix: bool = False, model: ModelEnum = ModelEnum.MBART50):
        print("Initializing Metrics")
        self.detection_keys = [
            'Detection Accuracy',
            'Detection Recall',
            'Detection Precision',
            'Detection F1',
            'Detection F0.5'
        ]
        self.correction_keys = [
            'Correction Accuracy',
            'Correction Recall',
            'Correction Precision',
            'Correction F1',
            'Correction F0.5'
        ]
        self.final_results = {
            'Detection Accuracy': 0,
            'Detection Recall': 0,
            'Detection Precision': 0,
            'Detection F1': 0,
            'Detection F0.5': 0,
            'Correction Accuracy': 0,
            'Correction Recall': 0,
            'Correction Precision': 0,
            'Correction F1': 0,
            'Correction F0.5': 0,
        }

        self.loss = 0
        self.ZWJ_Fix = True if model in [ModelEnum.MBART50, ModelEnum.MT5_BASE, ModelEnum.MT5_XL] else False
        self.model = model
        self.original_sentences = []
        self.predicted_sentences = []
        self.expected_sentences = []
        self.data_loader_length = len(data_loader)
        self.exp_dir = exp_dir
     
    def add_batch(self, original, predictions, labels):
        self.original_sentences.extend(original)
        self.predicted_sentences.extend(predictions)
        self.expected_sentences.extend(labels)   
    
    def evaluate(
        self, 
        gathered_original: torch.Tensor, 
        gathered_predictions: torch.Tensor, 
        gathered_labels: torch.Tensor, 
        tokenizer, 
        device: torch.device, 
        epoch: str = "Test",
        replace: bool = False
        ):
        
        skip_special_tokens = (self.ZWJ_Fix == False)
        if self.ZWJ_Fix:
            special_token_to_keep = "<ZWJ>"
            special_token_id_to_keep = tokenizer.convert_tokens_to_ids(special_token_to_keep)
            if self.model == ModelEnum.MBART50:
                mbart_lang_code = tokenizer.lang_code_to_id["si_LK"]
                all_special_ids = torch.tensor(tokenizer.all_special_ids + [mbart_lang_code], dtype=torch.int64).to(device)
            elif self.model == ModelEnum.XLMR_BASE:
                seperator_code = tokenizer.sep_token_id
                all_special_ids = torch.tensor(tokenizer.all_special_ids + [seperator_code], dtype=torch.int64).to(device)
            else:
                all_special_ids = torch.tensor(tokenizer.all_special_ids, dtype=torch.int64).to(device)
                
            special_token_id_to_keep_tensor = torch.tensor([special_token_id_to_keep], dtype=torch.int64).to(device)
            gathered_original = [PrePostProcessingUtils.remove_special_tokens(tokens.to(device), all_special_ids, special_token_id_to_keep_tensor) for tokens in gathered_original]
            gathered_predictions = [PrePostProcessingUtils.remove_special_tokens(tokens.to(device), all_special_ids, special_token_id_to_keep_tensor) for tokens in gathered_predictions]
            gathered_labels = [PrePostProcessingUtils.remove_special_tokens(tokens.to(device), all_special_ids, special_token_id_to_keep_tensor) for tokens in gathered_labels]
        
        eval_results = EvaluateUtils.evaluate(
            [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '')
            # .replace(" ", "")
            .replace("#", "")
            .replace(":", "")
            .strip() for ids in gathered_original],
            [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in gathered_predictions],
            [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in gathered_labels], 
            replace
        )        
        for key in self.final_results:
            self.final_results[key] += eval_results[key]

        self.add_batch(gathered_original, gathered_predictions, gathered_labels)
        
    def end_epoch(self, epoch, val_loss, train_loss, t):
        for key in self.final_results:
            self.final_results[key] /= self.data_loader_length
            
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        d_metrics = [f'{key} : {self.final_results[key]}' for key in self.detection_keys]
        c_metrics = [f'{key} : {self.final_results[key]}' for key in self.correction_keys]
        print(f'Detection Metrics: {d_metrics}')
        print(f'Correction Metrics: {c_metrics}')
    
    def log_test(self, total_loss, tokenizer, ZWJ_Fix = False):
        from Utils.GoogleUtils import GoogleUtils

        skip_special_tokens = not ZWJ_Fix
        for key in self.final_results:
            self.final_results[key] /= self.data_loader_length
        
        d_metrics = [f'{key} : {self.final_results[key]}' for key in self.detection_keys]
        c_metrics = [f'{key} : {self.final_results[key]}' for key in self.correction_keys]
        self.print_results(self.final_results, self.final_results)
        
        original_decoded = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '')
            # .replace(" ", "")
            .replace("#", "")
            .replace(":", "")
            .strip() for ids in self.original_sentences]
        predictions_decoded = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '').strip() for ids in self.predicted_sentences] 
        labels_decoded = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '').strip() for ids in self.expected_sentences]

        # Create results data dictionary
        results_data = {
            'Original': original_decoded,
            'Corrected': predictions_decoded,
            'Expected': labels_decoded,
        }
        # Write CSV inside the main process condition
        results_df = pd.DataFrame(results_data)
        
        # When Using Bert models these character were present in the output. They throws errors when saving in excel format
        results_df = results_df.applymap(lambda x: x.replace('.\uffff', '').replace('\x11', '') if isinstance(x, str) else x)

        if ZWJ_Fix:
            print("Replacing ZWJ token")
            # for whatever reason the ZWJ token is decoded with a additional space in the end.
            results_df['Original'] = results_df['Original'].replace('<ZWJ> ', '\u200D', regex=True)
            results_df['Corrected'] = results_df['Corrected'].replace('<ZWJ> ', '\u200D', regex=True)
            results_df['Expected'] = results_df['Expected'].replace('<ZWJ> ', '\u200D', regex=True)
            results_df['Original'] = results_df['Original'].replace('<ZWJ>', '\u200D', regex=True)
            results_df['Corrected'] = results_df['Corrected'].replace('<ZWJ>', '\u200D', regex=True)
            results_df['Expected'] = results_df['Expected'].replace('<ZWJ>', '\u200D', regex=True)
        print(results_df.head(3))
        results_df.to_excel(f'{self.exp_dir}/test_results.xlsx', index=False)
        results_df.to_csv(f'{self.exp_dir}/test_results.csv', index=False)
        
        current_path = os.getcwd()

        google_utils = GoogleUtils()
        google_utils.upload_folder(f'{current_path}/{self.exp_dir}')

    def print_results(self, eval_results, eval_with_replace):
        with open(f'{self.exp_dir}/Results-{self.exp_dir}.txt', 'w') as log_file:
            for metric, value in eval_results.items():
                log_file.write(f"{metric}: {value}\n")
        
        df_eval = pd.DataFrame({
            'Metric': eval_with_replace.keys(),
            'With Replacement': [eval_with_replace[metric] for metric in eval_with_replace.keys()],
            'Without Replacement': [eval_with_replace[metric] for metric in eval_with_replace.keys()] # other wise print_in_tab_seperated_format breaks 2 lazy to fix it
        })
        
        print("------------------------------------------------------------------------------------------------------------")
        print(df_eval.to_string(index=False))
        print("------------------------------------------------------------------------------------------------------------")
        
        EvaluateUtils.print_in_tab_seperated_format(df_eval)