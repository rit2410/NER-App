# Named Entity Recognition (NER) – Multi-Model Streamlit App

An interactive **Streamlit application** that performs Named Entity Recognition (NER) using **state-of-the-art Transformer models** (BERT, RoBERTa, XLM-RoBERTa, and more).  
Users can upload text or PDFs, select one or multiple models, visualize entities, compare model outputs side-by-side, and export results.

---

## Features

| Feature | Description |
|----------|-------------|
| **Model Comparison Mode** | Run and compare multiple models (BERT, RoBERTa, etc.) side-by-side using interactive tabs |
| **Attention / Heatmap Visualization** | View token-level importance derived from entity scores for interpretability |
| **Nested Entity Support** | Handles overlapping entities (e.g., “University of California” within “California”) |
| **CSV Export** | Download entity results per model or combined across all models |
| **PDF/Text Input** | Upload `.txt` or `.pdf` files, or paste text directly |
| **Custom Hugging Face Model Support** | Use any token-classification model from Hugging Face Hub by entering its model ID |
| **User-Friendly Sidebar** | Control model, aggregation, device, and chunking parameters |
| **Caching** | Re-runs are near-instant thanks to model and tokenizer caching |

---

## Tech Stack

- **Streamlit** – UI & interactivity  
- **Transformers (Hugging Face)** – Model loading, tokenization, inference  
- **PyTorch** – Model backend  
- **Plotly** – Token-importance heatmaps  
- **PDFPlumber** – Extracts text from PDF uploads  
- **Pandas / NumPy** – Data handling  
- **Python 3.10+** recommended

---

## Project Structure

```plaintext
ner_app/
├── app.py                        # Streamlit main app
├── utils/
│   ├── inference.py              # Model loading, tokenization, inference, caching
│   ├── preprocessing.py          # Read text/PDF and clean text
│   └── visualization.py          # Render entities, dataframes, token heatmaps
├── data/
│   └── sample_dataset.txt        # (Optional) Example input file
├── requirements.txt
└── README.md
```

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
## How to Use
- Upload a .txt or .pdf file, or paste text in the sidebar.
- Select one or more models (e.g., BERT, RoBERTa, XLM-RoBERTa).
- (Optional) Enter any Hugging Face model ID for custom models, e.g.:
    1. dslim/bert-base-NER
    2. Jean-Baptiste/roberta-large-ner-english
    3. xlm-roberta-large-finetuned-conll03-english
- Adjust:
    1. Aggregation Strategy → simple, max, first, average
    2. Device → auto (GPU if available) or cpu
    3. Chunk Size → for long text documents
- Click Run NER.
- Explore:
    1. Entity table and highlights
    2. Token importance (heatmap)
    3. CSV download buttons
    4. Comparison tabs for each model
- Supported Models
    1. Model	Hugging Face ID
    2. BERT	dslim/bert-base-NER
    3. RoBERTa	Jean-Baptiste/roberta-large-ner-english
    4. XLM-RoBERTa	xlm-roberta-large-finetuned-conll03-english
    5. Custom Model	(User-provided HF ID)
- Exporting Results
    1. After inference, you can:
       - Download per-model CSV with entities (text, label, score, start, end)
       - Download a combined CSV (includes results from all models with a model column)
- Visualization Examples
    1. Type	Description
    2. Entity Highlights	Colored spans for each recognized entity
    3. Token Importance Bar	Proxy attention plot showing token relevance
    4. Model Tabs	Side-by-side comparison of BERT vs RoBERTa vs XLM-RoBERTa
 
---
       
## Example Input Text
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California in April 1976.
The company’s headquarters, known as Apple Park, is located in Cupertino, part of the San Francisco Bay Area.
Tim Cook became the CEO of Apple after Steve Jobs resigned in August 2011.
In November 2023, Apple partnered with the University of California, Berkeley to research advanced AI systems.

---

## Future Enhancements
- Add multilingual NER models (e.g., mBERT, XLM-RoBERTa large multilingual)
- Integrate sentence segmentation for long documents
- Add performance benchmarking across models
- Add per-entity confidence visualizations
