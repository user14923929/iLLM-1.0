# iLLM-1.0

![logo](.github/assets/logo.svg)

Personal local LLM. Open Source under MIT License.

## Requirements
- Python 3.10+
- torch
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to use

1. Create dataset:
```bash
python create_dataset.py
```
2. Train model and generate text:
```bash
python iLLM-1.0.py
```

## Files
`iLLM-1.0.py` — model, training, text generation
`create-dataset.py` — generates dataset
`requirements.txt` — dependencies
`LICENSE` — MIT License