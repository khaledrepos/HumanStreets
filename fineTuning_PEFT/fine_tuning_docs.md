# Fine-Tuning Documentation: Saudi Dialect QLoRA

This document provides an in-depth explanation of the fine-tuning process used to adapt the **LiquidAI LFM2-1.2B** model to the Saudi Dialect using the **SauDial** dataset.

## 1. Overview
We are using **QLoRA (Quantized Low-Rank Adaptation)**, a parameter-efficient fine-tuning (PEFT) technique. This allows us to fine-tune a large language model on consumer hardware by freezing the original model weights and training only a small set of adapter layers on top of them, while quantized to 4-bit precision.

### Key Technologies
*   **Hugging Face Transformers**: For model loading and tokenization.
*   **PEFT (Parameter-Efficient Fine-Tuning)**: For integrating LoRA adapters.
*   **ALIBI / bitsandbytes**: For 4-bit quantization (reducing memory usage).
*   **TRL (Transformer Reinforcement Learning)**: Provides the `SFTTrainer` for supervised fine-tuning.

## 2. The Model: LiquidAI/LFM2-1.2B
We are using the `LiquidAI/LFM2-1.2B` model.
*   **Architecture**: It is a 1.2 Billion parameter causal language model.
*   **Tokenizer**: Uses the same tokenizer as the base model. We ensure the `pad_token` is set to the `eos_token` to handle batching correctly.

## 3. The Dataset: SauDial
The dataset contains English text paired with its translation into various Saudi Dialects (features: `Dialect`, `English Text`, `Dialect Translation`).

**Source Reference**:
> SauDial Dataset: [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S2352340925006304#cebibl1)

### Data Formatting
To train the model to translate english to specific Saudi dialects, we format the data into **Instruction-Input-Response** pairs:

```text
### Instruction:
Translate the following English text to Saudi Dialect ({Dialect}).
Context: {In-Game Context}

### Input:
{English Text}

### Response:
{Dialect Translation}
```

This structure teaches the model to look at the instruction and input, then generate the correct dialectal response.

## 4. Technical Implementation (`finetuning_QLora.py`)

### A. Quantization Configuration (`BitsAndBytesConfig`)
To fit the model in memory, we load it in 4-bit precision using `bitsandbytes`.
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```
*   `load_in_4bit=True`: Compresses weights to 4-bit.
*   `bnb_4bit_compute_dtype=torch.float16`: Performs calculations in float16 for speed and stability.

### B. LoRA Configuration (`LoraConfig`)
We define the LoRA adapter settings:
```python
lora_config = LoraConfig(
    r=8,                      # Rank of the low-rank matrices (higher = more parameters)
    lora_alpha=16,            # Scaling factor
    target_modules=["q_proj", "v_proj"], # Which layers to attach adapters to (Attention layers)
    task_type="CAUSAL_LM"
)
```
This adds a small number of trainable parameters (adapters) to the attention query and value projections.

### C. Dataset Loading & Formatting
We rely on `pandas` to read the Excel file and a custom function to format rows into the prompt structure described above.
**Important Fix**: We manually apply the formatting to create a text column *before* passing it to the trainer to avoid compatibility issues with `trl`'s formatting pipeline.

### D. Training Configuration (`SFTConfig`)
Settings for the training loop:
*   `max_length=512`: Truncates sequences longer than 512 tokens.
*   `gradient_accumulation_steps=8`: Simulates a larger batch size by accumulating gradients over multiple steps.
*   `fp16=True`: Uses mixed precision training.
*   `learning_rate=2e-4`: Standard QLoRA learning rate.

### E. Trainer (`SFTTrainer`)
The `SFTTrainer` manages the training loop, data loading, and optimization. We pass the `SFTConfig` and the pre-formatted dataset (specifying `dataset_text_field="text"`).

## 5. How to Run
Prerequisites: `uv`, `openpyxl`, `pandas`.

```powershell
uv run --with openpyxl --with pandas python .\finetuning_QLora.py
```

## 6. Training Process
1.  **Loading**: Model is loaded in 4-bit.
2.  **Dataset Prep**: Excel is read, rows are formatted into prompts.
3.  **Initialization**: LoRA adapters are attached.
4.  **Training**: The model iterates through the dataset, optimizing *only* the LoRA parameters to minimize the difference between its generated translation and the reference response.
5.  **Saving**: Checkpoints are saved to `./saudi-lora-test`.

## 7. Performance Considerations
*   **Memory**: 4-bit quantization allows training on consumer GPUs (e.g., 6GB-8GB VRAM).
*   **Speed**: Training 1 epoch on a small dataset like this should take a few minutes.
