import os
import torch
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from Utils.DatasetUtils import DatasetUtils
from Trainer.AbstractTrainer import AbstractTrainer

class UnslothTrainer(AbstractTrainer):
    def __init__(
            self,
            model_instance, 
            train_path, 
            val_path,
            exp_name,
            batch_size = 4,
            epochs = 1,
            gradient_accumulation_steps = 2,
            lr = 1e-4,
            saving_steps=100000,
            dataset_size = 1, 
            load_in_4bit=True,
        ):

        self.model_instance = model_instance
        self.train_path = train_path
        self.val_path = val_path
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.saving_steps = saving_steps
        self.dataset_size = dataset_size
        self.dtype = torch.bfloat16
        self.load_in_4bit = load_in_4bit


    def train(self):
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        model_instance = self.model_instance

        if torch.cuda.get_device_name(0).startswith('Tesla T4'):
            self.dtype = torch.float16

        exp_name = self.exp_name
        dataset = DatasetUtils(train_path=self.train_path, val_path=self.val_path, dataset_size=self.dataset_size)

        def data_collator(examples):

            EOS_TOKEN = model_instance.tokenizer.eos_token
            texts = [model_instance.PROMPT.format(example['text'], example['expected']) + EOS_TOKEN for example in examples]

            batch = model_instance.tokenizer(texts, return_tensors="pt", padding=True)
            if batch["input_ids"].size(1) > model_instance.max_seq_length:
                batch["input_ids"] = batch["input_ids"][:, :model_instance.max_seq_length]
                batch["attention_mask"] = batch["attention_mask"][:, :model_instance.max_seq_length]

            labels = batch["input_ids"].clone()
            # Update the labels in the batch
            labels[labels == model_instance.tokenizer.pad_token_id] = -100
            batch["labels"] = labels   

            return batch

        training_args = SFTConfig(
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,  
            gradient_accumulation_steps = self.gradient_accumulation_steps,
            warmup_steps = 5,
            num_train_epochs = self.epochs,
            learning_rate = self.lr,
            report_to = 'none',
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1000,  # Logs every 1000 steps
            eval_steps = self.saving_steps,  # Evaluate every 10,000 steps
            eval_strategy = "steps",  # Perform evaluation based on steps
            save_steps = self.saving_steps,
            save_total_limit = 4,  # Keep only the last 4 checkpoints
            load_best_model_at_end = True,  # Load the best model after training ends
            metric_for_best_model = "eval_loss",  # Use eval_loss for early stopping
            gradient_checkpointing = True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            output_dir = exp_name,
            seed = model_instance.seed,
            remove_unused_columns = False,
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_text_field = "text",
            max_seq_length = model_instance.max_seq_length,
            dataset_num_proc = os.cpu_count(),
        )
        
        trainer = SFTTrainer(
            model = model_instance.model,
            tokenizer = model_instance.tokenizer,
            data_collator = data_collator,
            train_dataset = dataset.train_dataset,
            eval_dataset = dataset.val_dataset,
            args = training_args,
        )

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        trainer.train()
        torch.cuda.empty_cache()

        model_instance.model.save_pretrained(exp_name)
        model_instance.tokenizer.save_pretrained(exp_name)
        torch.save(trainer.optimizer.state_dict(), os.path.join(exp_name, "optimizer.pt"))
        torch.save(trainer.lr_scheduler.state_dict(), os.path.join(exp_name, "scheduler.pt"))
        print("Model saved successfully.")

        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()