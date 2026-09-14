# LMSpell

**LMSpell** is a flexible and extensible library designed to facilitate the implementation of spell correction systems using a wide range of pre-trained language models (PLMs). It supports encoder-based, decoder-based (i.e. LLMs), and encoder-decoder-based models, providing broad compatibility across diverse architectures.

The interface of LMSpell, abstracts core PLM functionalities such as fine-tuning and inference. This abstraction enables users to integrate and experiment with different model types without requiring model-specific implementation knowledge. Internally, LMSpell dynamically loads the appropriate classes based on the selected model, ensuring a consistent and user-friendly development experience.

## Installation

Library will be publicaly released upon the acceptance of the paper

```bash
pip install LMSpell
```

## Quick Start

### 1. Initialize the Model

```python
from LMSpell import LMSpell

# Initialize the LMSpell instance
model_instance = LMSpell("unsloth/gemma-2-9b-bnb-4bit")
```

### 2. Fine-tuning with a Trainer

A trainer class can be used to fine-tune a model. Three trainers have been provided for decoder-based, encoder-decoder-based and encoder-based models.

#### a. Decoder-based

```python
from LMSpell.Trainer.UnslothTrainer import UnslothTrainer

trainer = UnslothTrainer(
    model_instance=model_instance,
    train_path="train.csv",
    val_path="val.csv",
    exp_name="demo",
    language="en"
)
```
#### b. Encoder-decoder-based

```python
from LMSpell.Trainer.Seq2SeqTrainer import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model_instance=model_instance,
    train_path="train.csv",
    val_path="val.csv",
    exp_name="demo",
    language="en"
)
```

#### c. Encoder-based

```python
from LMSpell.Trainer.SeqLabelTrainer import SeqLabelTrainer

trainer = SeqLabelTrainer(
    model_instance=model_instance,
    train_path="train.csv",
    val_path="val.csv",
    exp_name="demo",
    language="en"
)
```

Call train method on trainer to fine-tune the model

```python
trainer.train()
```

### 3. Run Inference

```python
import pandas as pd

test_set = pd.read_csv("test.csv")
model_instance.correct(test_set)
```

##  Supported Models

#### Fine-tuned Checkpoints  
*Note: These checkpoints will be released upon acceptance of the paper.*

#### LLaMA 3.1
- `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`

#### LLaMA 3.2
- `unsloth/Llama-3.2-1B-bnb-4bit`
- `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- `unsloth/Llama-3.2-3B-bnb-4bit`
- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`

#### Gemma 2
- `unsloth/gemma-2-2b-bnb-4bit`
- `unsloth/gemma-2-2b-it-bnb-4bit`
- `unsloth/gemma-2-9b-bnb-4bit`
- `unsloth/gemma-2-9b-it-bnb-4bit`


#### mT5
- `google/mt5-small`
- `google/mt5-base`
- `google/mt5-large`
- `google/mt5-xl`
- `google/mt5-xxl`

#### MBART
- `facebook/mbart-large-50`

#### SinBERT (Sinhala)
- `NLPC-UOM/SinBERT-small`
- `NLPC-UOM/SinBERT-large`

#### XLM-R
- `FacebookAI/xlm-roberta-base`
- `FacebookAI/xlm-roberta-large`

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  