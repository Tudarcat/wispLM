---
languages:
    - zh
    - en
    - multilingual
tags:
    - tokenizer
    - bilingual
    - chinese
    - english
    - multilingual
license: apache-2.0
---
# QiYuanTokenizer-Small

**QiYuanTokenizer** is a *universal multilingual tokenizer* primarily optimized for **Chinese–English mixed text**,  
offering compact and efficient tokenization across diverse languages and scripts.  
It is designed as a **general-purpose tokenizer**, not tied to any specific model family,  
and is especially suitable for **encoder** and **encoder-decoder** architectures.

---

## ✨ Overview

| Property                | Value                               |
|-------------------------|-------------------------------------|
| **Name**                | QiYuanTokenizer-Small               |
| **Type**                | Tokenizer-only repository           |
| **Purpose**             | General multilingual tokenization   |
| **Primary Languages**   | Chinese, English                    |
| **Extended Support**    | Multilingual (Unicode-complete)     |
| **Architecture**        | Unigram                             |
| **Vocabulary Size**     | 24,000 tokens                       |
| **Fast Implementation** | ✅ Available (`QiYuanTokenizerFast`) |
| **Framework**           | 🤗 `transformers`                   |
| **License**             | Apache 2.0                          |

---

## 🧩 QiYuan Tokenizer Series

| Variant                                                                               | Vocabulary Size | Description                                                                                                                                 | Recommended Use                                    |
|---------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| [**QiYuanTokenizer-Tiny**](https://huggingface.co/Morton-Li/QiYuanTokenizer-Tiny)     | 12k             | Extremely compact vocabulary for highly constrained settings. Efficient, but may become limiting for more demanding multilingual scenarios. | **Use with caution**                               |
| [**QiYuanTokenizer-Small**](https://huggingface.co/Morton-Li/QiYuanTokenizer-Small)   | 24k             | A lightweight tokenizer with improved coverage over Tiny while still keeping vocabulary size modest.                                        | Compact models and efficiency-oriented experiments |
| [**QiYuanTokenizer-Base**](https://huggingface.co/Morton-Li/QiYuanTokenizer-Base)     | 32k             | A balanced baseline vocabulary suitable for general bilingual and multilingual tokenization tasks.                                          | **Recommended for general use**                    |
| [**QiYuanTokenizer-Medium**](https://huggingface.co/Morton-Li/QiYuanTokenizer-Medium) | 48k             | The best-balanced variant in the series, providing strong coverage and good compression while keeping model complexity reasonable.          | **Recommended balance choice**                     |
| [**QiYuanTokenizer-Large**](https://huggingface.co/Morton-Li/QiYuanTokenizer-Large)   | 64k             | A larger vocabulary designed for quality-oriented training, offering better coverage and stronger tokenization fidelity.                    | **Recommended when quality is prioritized**        |

> All variants share the same core token definitions and compatible special token settings.

---

## ⚙️ Usage

You can load this tokenizer directly with `AutoTokenizer`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Morton-Li/QiYuanTokenizer-Small", trust_remote_code=True)

# Example
text = "你好，QiYuan！"
tokens = tokenizer(text)
print(tokens["input_ids"])
```

### ➕ Batch Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Morton-Li/QiYuanTokenizer-Small", trust_remote_code=True)

# Example
texts = [
    "Hello, 世界！",
    "QiYuanTokenizer is designed for multilingual tokenization."
]
batch_tokens = tokenizer(texts, padding=True, return_tensors="pt")
print(batch_tokens["input_ids"])
```

---

## 🧠 Design Notes

QiYuanTokenizer adopts the **Unigram** algorithm and is intended as a practical tokenizer for **general text understanding and sequence transformation tasks**.

In practice, it is generally more suitable for:

* **Encoder models**, such as text classification, embedding, retrieval, and sequence labeling
* **Encoder-decoder models**, such as translation, summarization, and text transformation

It can still be used in broader settings, but its design is not primarily oriented toward chat-format tokenization or decoder-only conversational templates.

---

## 📦 Files Included

| File                    | Description                                                      |
|-------------------------|------------------------------------------------------------------|
| `tokenizer.json`        | Serialized fast tokenizer definition                             |
| `tokenizer_config.json` | Configuration (max length, padding side, etc.)                   |
| `tokenizer.py`          | Tokenizer implementation                                         |
| `tokenizer.model`       | SentencePiece model file trained with the Unigram algorithm      |
| `tokenizer.vocab`       | SentencePiece vocabulary file corresponding to `tokenizer.model` |

---

## 🔍 Special Tokens

| Token        | Purpose                               |
|--------------|---------------------------------------|
| `<\|unk\|>`  | Unknown token                         |
| `<\|bos\|>`  | Beginning of sequence                 |
| `<\|eos\|>`  | End of sequence                       |
| `<\|pad\|>`  | Padding token for batch alignment     |
| `<\|mask\|>` | Masked token for MLM-style objectives |

---

## 🔖 License

This tokenizer and vocabulary are released under the **Apache License 2.0**.
You are free to use, modify, and redistribute it under the same license terms.

---

## 📚 Citation

If you use **QiYuanTokenizer** in your research or project, please cite it as:

```bibtex
@misc{QiYuanTokenizer,
  title  = {QiYuanTokenizer: A Universal Multilingual Unigram Tokenizer with Chinese-English Optimization},
  author = {Morton Li},
  year   = {2026},
}
```
