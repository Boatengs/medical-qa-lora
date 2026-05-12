#  Medical QA LLM — LoRA Fine-tuning

Fine-tuned Mistral 7B on 5,000 medical QA examples using LoRA + 4-bit QLoRA.
Trained entirely for free on Google Colab T4 GPU.

[![HuggingFace](https://img.shields.io/badge/%20Model-HuggingFace%20Hub-yellow)](https://huggingface.co/samurvivor-07/medical-mistral-lora)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab-orange)](https://colab.research.google.com)

##  Results

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 100  | 0.6992    | 0.6814   |
| 200  | 0.6724    | 0.6647   |
| 300  | 0.6792    | 0.6573   |

##  Architecture

| Component | Choice |
|-----------|--------|
| Base Model | Mistral-7B-Instruct-v0.3 |
| Fine-tuning | LoRA (rank=16, alpha=32) |
| Quantization | 4-bit QLoRA via bitsandbytes |
| Dataset | Medical Meadow Flashcards (5,000 examples) |
| Trainable Params | 13.6M / 3.7B (0.36%) |
| Hardware | T4 GPU — Google Colab Free |
| Cost | $0 |

##  Quick Start

```python

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
adapter    = "samurvivor-07/medical-mistral-lora"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model      = PeftModel.from_pretrained(model, adapter)

prompt = """### Instruction:
Answer this question truthfully

### Input:
What is the mechanism of action of beta blockers?

### Response:
"""

inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

##  Sample Outputs

**Q: What is the mechanism of action of beta blockers?**
> Beta blockers work by blocking beta-adrenergic receptors, reducing heart rate, lowering blood pressure, and improving blood flow. Used for hypertension, angina, and heart failure.

**Q: How does aspirin prevent blood clots?**
> Aspirin prevents blood clots by inhibiting platelet aggregation through COX inhibition.

**Q: What are the symptoms of diabetic ketoacidosis?**
> Symptoms include nausea, vomiting, and abdominal pain.

##  Key Technical Decisions

**Why LoRA over full fine-tuning?**
Only 0.36% of parameters are trainable (13.6M / 3.7B), making it feasible on free hardware with no quality loss.

**Why 4-bit quantization?**
Reduces model from 14GB to 3.5GB VRAM — enabling free T4 GPU training.

**Why Mistral 7B?**
Ungated, strong medical reasoning, outperforms Llama 3.2 3B on QA tasks.

##  Limitations

- Trained on flashcard-style QA only
- Not validated for clinical use
- Should not replace professional medical advice

##  Links

- [Model on HuggingFace Hub](https://huggingface.co/samurvivor-07/medical-mistral-lora)
- [Project 1 — Sentiment Analyzer](https://github.com/Boatengs/sentiment-analyzer)
- [Project 2 — SPORTZBOT RAG Chatbot](https://github.com/Boatengs/sports-rag-chatbot-)
