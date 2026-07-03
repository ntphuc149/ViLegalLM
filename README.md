# [ViLegalLM: Language Models for Vietnamese Legal Text](https://aclanthology.org/2026.findings-acl.1801/)

<p align="center">
  <a href="https://github.com/ntphuc149/ViLegalLM"><img src="https://img.shields.io/badge/GitHub-ViLegalLM-blue?logo=github" /></a>
  <a href="https://huggingface.co/ntphuc149"><img src="https://img.shields.io/badge/🤗-Models%20%26%20Datasets-yellow" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" /></a>
</p>

This repository contains the source code for **ViLegalLM**, a suite of language models pre-trained on Vietnamese legal text. Models and datasets are hosted on Hugging Face.

> **Note:** These models are intended for research purposes only and should not be used in production legal applications without expert validation.

---

## Models

| Model                    | Architecture           | Params | Context      | HuggingFace                                                                                     |
| ------------------------ | ---------------------- | ------ | ------------ | ----------------------------------------------------------------------------------------------- |
| ViLegalBERT              | Encoder-only (RoBERTa) | 135M   | 256 tokens   | [ntphuc149/ViLegalBERT](https://huggingface.co/ntphuc149/ViLegalBERT)                           |
| ViLegalQwen2.5-1.5B-Base | Decoder-only (Qwen2)   | 1.54B  | 2,048 tokens | [ntphuc149/ViLegalQwen2.5-1.5B-Base](https://huggingface.co/ntphuc149/ViLegalQwen2.5-1.5B-Base) |
| ViLegalQwen3-1.7B-Base   | Decoder-only (Qwen3)   | 1.72B  | 4,096 tokens | [ntphuc149/ViLegalQwen3-1.7B-Base](https://huggingface.co/ntphuc149/ViLegalQwen3-1.7B-Base)     |

All models are pre-trained on a 16GB Vietnamese legal corpus ([ntphuc149/ViLegalText](https://huggingface.co/datasets/ntphuc149/ViLegalText)).

### Loading Models

**ViLegalBERT**

> Usable out-of-the-box for feature extraction and semantic similarity. Fine-tuning required for downstream tasks (NLI, classification, extractive QA).
>
> **Important:** Input text must be **word-segmented** before feeding into the model. Use [PyVi](https://github.com/trungtv/pyvi), [underthesea](https://github.com/undertheseanlp/underthesea), or [VNCoreNLP](https://github.com/vncorenlp/VNCoreNLP). Words in a multi-syllable token should be joined by underscores (e.g., `nghiên_cứu_viên`). See [PhoBERT](https://github.com/VinAIResearch/PhoBERT#-using-phobert-with-transformers) for details.

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("ntphuc149/ViLegalBERT")

# Input text MUST be word-segmented (multi-syllable words joined by underscores)
sentence = "luật_sư tư_vấn pháp_luật dân_sự ."

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    features = model(input_ids)
```

**ViLegalQwen2.5-1.5B-Base** and **ViLegalQwen3-1.7B-Base**

> These are base models (CLM). Fine-tuning or instruction tuning (QLoRA) is required before using for downstream tasks.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# ViLegalQwen2.5-1.5B-Base
tokenizer = AutoTokenizer.from_pretrained("ntphuc149/ViLegalQwen2.5-1.5B-Base")
model = AutoModelForCausalLM.from_pretrained("ntphuc149/ViLegalQwen2.5-1.5B-Base")

# ViLegalQwen3-1.7B-Base
tokenizer = AutoTokenizer.from_pretrained("ntphuc149/ViLegalQwen3-1.7B-Base")
model = AutoModelForCausalLM.from_pretrained("ntphuc149/ViLegalQwen3-1.7B-Base")
```

---

## Datasets

Synthetic datasets generated via LLM generation and negative sampling.

| Dataset    | Task                       | Train  | Val | HuggingFace                                                                  |
| ---------- | -------------------------- | ------ | --- | ---------------------------------------------------------------------------- |
| ViLegalTF  | True/False QA              | 13,032 | 388 | [ntphuc149/ViLegalTF](https://huggingface.co/datasets/ntphuc149/ViLegalTF)   |
| ViLegalMCQ | Multiple-Choice QA         | 14,920 | 300 | [ntphuc149/ViLegalMCQ](https://huggingface.co/datasets/ntphuc149/ViLegalMCQ) |
| ViLegalNLI | Natural Language Inference | 7,660  | 150 | [ntphuc149/ViLegalNLI](https://huggingface.co/datasets/ntphuc149/ViLegalNLI) |

---

## Quick Start

### Pre-training

All pre-training code is in `Source codes/Pre-training ViLegalLM/`:

- `ViLegalBERT/` — MLM pre-training scripts and configs (PhoBERT-based)
- `ViLegalQwen/` — CLM pre-training scripts and configs (Qwen-based)

### Fine-tuning

All fine-tuning code is provided as Jupyter Notebooks in `Source codes/Fine-tuning ViLegalLM/`.

**Notation:**

- `[FT]` — Discriminative fine-tuning (encoder models)
- `[IFT]` — Instruction fine-tuning with QLoRA (decoder models)
- `[FT-CV]` — 5-fold cross-validation

---

## Repository Structure

```
ViLegalLM/
└── Source codes/
    ├── Pre-training ViLegalLM/     # Pre-training scripts and configs
    │   ├── ViLegalBERT/            # MLM pre-training (PhoBERT-based)
    │   └── ViLegalQwen/            # CLM pre-training (Qwen-based)
    └── Fine-tuning ViLegalLM/      # Fine-tuning notebooks (.ipynb)
        ├── Information Retrieval
        ├── Question Answering      # True/False, MCQ, Extractive, Abstractive
        ├── Natural Language Inference
        └── Syllogism Reasoning
```

---

## Citation

```bibtex
@inproceedings{nguyen-etal-2026-vilegallm,
    title = "{V}i{L}egal{LM}: Language Models for {V}ietnamese Legal Text",
    author = "Nguyen, Truong-Phuc  and
      Nguyen, Quy-Nhan  and
      Nguyen, Minh-Tien",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1801/",
    pages = "36136--36150",
    ISBN = "979-8-89176-395-1",
    abstract = "We present **ViLegalLM**, comprising **ViLegalBERT** and **ViLegalQwen**, the first suite of Vietnamese pretrained language models for legal text understanding and generation. It includes one encoder-only model (ViLegalBERT, 135M parameters) and two decoder-only models (ViLegalQwen2.5-1.5B-Base and ViLegalQwen3-1.7B-Base), all continually pretrained on a newly curated 16GB Vietnamese legal corpus, significantly larger than previous work. To mitigate data scarcity, we construct three synthetic datasets using LLM-based generation and hard negative mining for True/False QA, Multiple Choice QA, and Natural Language Inference. We establish state-of-the-art results among open-source models on four main Vietnamese legal downstream tasks spanning ten benchmarks, demonstrating that continual pretraining from base models consistently outperforms instruction-tuned adaptation. Source codes, corpus, datasets, and model checkpoints are publicly available at https://github.com/ntphuc149/ViLegalLM."
}
```
