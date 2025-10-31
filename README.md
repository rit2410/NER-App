# Multi-Model Named Entity Recognition (NER) App (with Debug Logging)

A **Streamlit-based interactive NER application** that allows users to upload text or PDF files, choose from multiple pre-trained Transformer models (BERT, RoBERTa, XLM-RoBERTa, or any Hugging Face model), and view extracted named entities in both table and color-highlighted formats.

This version includes **detailed print-based debugging** across all components so you can trace every step directly in your terminal — model loading, text chunking, inference, rendering, etc.

---

## Features

- **Input Sources**
  - Upload `.txt` or `.pdf` files  
  - Paste free-form text directly into the app

- **Multiple Models**
  - `dslim/bert-base-NER`
  - `Jean-Baptiste/roberta-large-ner-english`
  - `xlm-roberta-large-finetuned-conll03-english`
  - or any **custom Hugging Face** model fine-tuned for token classification

- **Debug Logging**
  - Every key operation (`load model`, `read file`, `chunk text`, `run inference`) prints status to the terminal
  - Helps identify slow or failing steps during first-time model loading

- **Results Display**
  - Color-highlighted entities in text view  
  - Interactive table with entity, label, confidence score, and offsets  
  - Expandable JSON view for raw model output

- **Device & Chunk Controls**
  - Select GPU/CPU automatically (`auto` or `cpu`)
  - Adjustable chunk size for long texts

---

## Project Structure

ner_app/
├── app.py # Main Streamlit UI with debug prints
├── utils/
│ ├── preprocessing.py # File reading & cleaning (TXT/PDF)
│ ├── inference.py # Model loading, pipeline setup, NER inference
│ └── visualization.py # Entity highlighting & table conversion
├── requirements.txt
└── README.md


---

## Installation

### 1. Create and activate environment
```bash
conda create -p ./venv python=3.11
conda activate ./venv
```

### 2. Install dependencies

- pip install --upgrade pip setuptools wheel
- pip install -r requirements.txt

If installation stalls or Torch is slow on macOS:

- conda install pytorch torchvision torchaudio cpuonly -c pytorch
- pip install -r requirements.txt --no-deps

---
  
## Deployed App

---
## Example Workflow
1. Select model in the sidebar (e.g., RoBERTa)
2. Upload text or PDF or paste raw text
3. Click Run NER
4. Observe progress in your terminal
5. View:
   - Entities Table with scores
   - Highlighted Text
   - Raw JSON output
6. Model	Description
- dslim/bert-base-NER	General English NER baseline
- Jean-Baptiste/roberta-large-ner-english	RoBERTa model fine-tuned on NER
- xlm-roberta-large-finetuned-conll03-english	Cross-lingual (XLM-R) variant
- dslim/distilbert-NER	Lightweight & faster baseline

---

## Future Enhancements
Add model comparison mode (side-by-side results)
Add attention/heatmap visualization
Include nested-entity handling
Save results to CSV

