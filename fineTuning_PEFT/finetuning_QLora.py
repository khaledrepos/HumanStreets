from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
import pandas as pd
from datasets import Dataset

model_id = "LiquidAI/LFM2-1.2B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,                      # SMALL
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],  # reduce memory
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load Excel dataset
df = pd.read_excel("SauDial Dataset.xlsx")

# Convert to HF Dataset
# dataset = Dataset.from_pandas(df)

def formatting_prompts_func(row):
    dialect = row["Dialect"]
    context = row["In-Game Context"]
    english = row["English Text"]
    response = row["Dialect Translation"]
    
    text = f"""### Instruction:
Translate the following English text to Saudi Dialect ({dialect}).
Context: {context}

### Input:
{english}

### Response:
{response}
"""
    return text

# Apply formatting manually to ensure 'text' column is strings
df["text"] = df.apply(formatting_prompts_func, axis=1)
dataset = Dataset.from_pandas(df)

# Ensure we have a train split (if not already split)
if "train" not in dataset:
    dataset = dataset.train_test_split(test_size=0.1)

# training_args = TrainingArguments(
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     num_train_epochs=1,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=5,
#     output_dir="./saudi-lora-test",
#     save_total_limit=1,
# )

sft_config = SFTConfig(
    output_dir="./saudi-lora-test",
    max_length=512,
    packing=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_total_limit=1,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    args=sft_config,
)

trainer.train()

