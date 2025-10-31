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
## Example Workflow
1. Select model in the sidebar (e.g., RoBERTa)
2. Upload text or PDF or paste raw text
3. To test the app try pasting below text -
   
*Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California in April 1976.
The company’s headquarters, known as Apple Park, is located in the city of Cupertino, part of the San Francisco Bay Area.

Tim Cook became the CEO of Apple after Steve Jobs resigned in August 2011. Under Cook’s leadership, Apple launched
several groundbreaking products including the iPhone 13 Pro, Apple Watch Series 9, and MacBook Air M3.

In November 2023, Apple partnered with the University of California, Berkeley to research advanced AI systems
for medical diagnostics. The project was co-funded by the U.S. Department of Health and the European Commission.

Meanwhile, Microsoft Corporation, based in Redmond, Washington, announced a collaboration with OpenAI
to integrate GPT-4 into Microsoft Office 365. This collaboration was first revealed at the Microsoft Build Conference
held in Seattle on May 22, 2023.

Elon Musk’s company, SpaceX, launched the Starship rocket from Boca Chica, Texas on April 20, 2023,
and later secured a contract with NASA worth $2.9 billion. NASA Administrator Bill Nelson congratulated SpaceX
on the successful launch, calling it “a new era in space exploration.”

Notably, the University of Oxford in England conducted a comparative study between Apple’s and Microsoft’s
AI ethics frameworks in 2024.
*

4. Click Run NER
5. Observe progress in your terminal
7. View:
   - Entities Table with scores
   - Highlighted Text
   - Raw JSON output
8. Model	Description
- dslim/bert-base-NER	General English NER baseline
- Jean-Baptiste/roberta-large-ner-english	RoBERTa model fine-tuned on NER
- xlm-roberta-large-finetuned-conll03-english	Cross-lingual (XLM-R) variant
- dslim/distilbert-NER	Lightweight & faster baseline

