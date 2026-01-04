---
base_model: LiquidAI/LFM2-1.2B
library_name: transformers
model_name: saudi-lora-test
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for saudi-lora-test

This model is a fine-tuned version of [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.24.0
- Transformers: 4.57.3
- Pytorch: 2.5.1+cu121
- Datasets: 4.4.2
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```