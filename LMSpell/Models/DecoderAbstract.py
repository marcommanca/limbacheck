import pandas as pd
from Utils.ConfigUtils import ConfigUtils
from Utils.EvaluateUtils import EvaluateUtils
from Models.ModelAbstract import ModelAbstract
from LMSpellException import LMSpellException

class DecoderAbstract(ModelAbstract):

    def __init__(self):
        import torch
        from unsloth import FastLanguageModel

        super().__init__()

        
        self.max_seq_length = 2048    # WHY!!!
        self.dtype = torch.float16 if torch.cuda.get_device_name(0).startswith('Tesla T4') else torch.bfloat16
        self.load_in_4bit = True    # Get as a parameter
        self.attn_implementation = "flash_attention_2"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_label.get_model_path(),
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            attn_implementation=self.attn_implementation,
            random_state = self.seed,
        )

        # Need to check pros and cons of loading it like this for both training and inference.
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8, # 8, 16, 32, 64, 128 suggested
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = self.seed,
            use_rslora = False,
            loftq_config = None,
        )

        self.model = model
        self.tokenizer = tokenizer

        self.PROMPT = f"""You are an expert {self.language} spell corrector. Below is a sentence in {self.language} language. It may or may not have a spelling mistake. Give the corrected output in {self.language}.

        ### Text:
        {{}}

        ### Output:
        {{}}"""
                
    def correct(self, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str | pd.DataFrame = None, output_dir: str = 'outputs'):
        """
        Corrects the text. The input set should be a string, a list of strings or a DataFrame.
        Perform the evaluation if the target set is provided.
        """
        import re        
        import os
        from tqdm import tqdm
        from unsloth import FastLanguageModel

        os.makedirs(os.path.join(os.getcwd(), output_dir), exist_ok=True)
        
        input_set, evaluate_flag = self.process_input(input_set, target_set)

        dataset_col_names = ConfigUtils.get_dataset_columns()
        
        FastLanguageModel.for_inference(self.model) 
        pred_texts = []
        generation_mode = self.model.generation_config.get_generation_mode()
        print(f"Generation mode: {generation_mode}")

        for src in tqdm(input_set[dataset_col_names[0]].to_list(), desc="Generating predictions"):
            inputs = self.tokenizer(
                [self.PROMPT.format(src, "")],
                return_tensors="pt"
            ).to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=self.max_seq_length, do_sample=self.do_sample, use_cache=True)

            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            pred_texts.append(pred_text)
        
        output_texts = []
        for pred in pred_texts:
            match = re.search(r'### Output:\s*([^\n]+)', pred)
            output_texts.append(match.group(1).strip() if match else None)

        result_col_names = ConfigUtils.get_results_columns()

        results_data = {
            result_col_names[0]: input_set[dataset_col_names[0]].to_list(),
            result_col_names[1]: output_texts
        }

        results_df = pd.DataFrame(results_data)

        if evaluate_flag:
            results_df[result_col_names[2]] = input_set[dataset_col_names[1]].to_list()
            print("Evaluating the outputs...")
            EvaluateUtils.evaluate_from_dataframe(results_df, output_dir)            

        return results_df

    def correctFromFile(self, src: str, target: str = None, output_dir: str = "outputs"):
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
                
        return self.correct(src, target, output_dir)
