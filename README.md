# Fine-Tune LM (GPT-2) with Custom Data

This repository demonstrates how to fine-tune a [GPT-2](https://huggingface.co/openai-community/gpt2) language model for dialog/chat tasks using your own dataset of prompt-response pairs. It supports training on CPU, CUDA GPU, or Apple Silicon (MPS). Any additions or deletions can be made to the `data.jsonl` file but the format must be followed.

## Features

- Fine-tunes GPT-2 or any HuggingFace-compatible causal LLM on a custom dataset.
- Supports training with very small to large datasets in `data.jsonl` format.
- Includes interactive test script for quick evaluation after training.

## Chat.py usage

Before running open the `chat.py` file in a text editor and change the `line 4(MODEL_NAME)` to latest model checkpoint inside the `results` folder, if no checkpoint is available tune the model first via `tune.py` and wait for the process to complete.

## Requirements

- Python 3.13+
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)

### Install dependencies:

*Before installing any dependencies create a python virtual environment for cleaner setup.*

```bash
pip install torch transformers datasets
```