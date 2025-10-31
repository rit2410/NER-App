# Named Entity Recognition (NER) and News Classification – Multi-Model Streamlit App

An interactive **Streamlit application** that performs both Named Entity Recognition (NER) and News Classification using **state-of-the-art Transformer models (BERT, RoBERTa, XLM-RoBERTa, DistilBERT, and others)**.
Users can upload text or PDFs, select one or multiple models, visualize entities or classification probabilities, compare model outputs side-by-side, and export results.

---

## Features

| Feature                    | Description                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| **Dual Task Support**      | Switch between *NER* and *News Classification* modes directly in the app                       |
| **Model Comparison Mode**  | Run and compare multiple models (BERT, RoBERTa, etc.) side-by-side using interactive tabs      |
| **Attention / Heatmap Visualization** | View token-level importance derived from entity scores for interpretability         |
| **Nested Entity Support**             | Handles overlapping entities (e.g., “University of California” within “California”) |
| **Classification Probabilities Bar**  | Visualizes category probabilities for each classification model |
| **CSV Export**                        | Download entity or classification results per model or combined across all models |
| **PDF/Text Input**                    | Upload `.txt` or `.pdf` files, or paste text directly  |
| **Custom Hugging Face Model Support** | Use any token-classification or sequence-classification model from the Hugging Face Hub by entering its model ID |
| **User-Friendly Sidebar**             | Control model, aggregation, device, and chunking parameters |
| **Caching**                           | Re-runs are near-instant thanks to model and tokenizer caching |

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
│   ├── inference.py              # Model loading, tokenization, inference, caching for NER
│   ├── classification.py         # Model loading, inference for news/topic classification
│   ├── preprocessing.py          # Read text/PDF and clean text
│   └── visualization.py          # Render entities, dataframes, token/classification heatmaps
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

https://ner-lab.streamlit.app

---
## How to Use
- Choose the task type in the sidebar:
    1. Named Entity Recognition (NER)
    2. News Classification
- Upload a .txt or .pdf file, or paste text in the sidebar.

### For NER
- Select one or more models
- Supported Models
  
| Model        | Hugging Face ID                             |
| ------------ | ------------------------------------------- |
| BERT         | dslim/bert-base-NER                         |
| RoBERTa      | Jean-Baptiste/roberta-large-ner-english     |
| XLM-RoBERTa  | xlm-roberta-large-finetuned-conll03-english |
| Custom Model | (User-provided HF ID)                       |

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

### For News Classification
- Supported Models
  
| Model                          | Hugging Face ID                                                   |
| ------------------------------ | ----------------------------------------------------------------- |
| DistilBERT (SST-2 Sentiment)   | distilbert-base-uncased-finetuned-sst-2-english                   |
| RoBERTa (Tweet Topic 21 Multi) | cardiffnlp/tweet-topic-21-multi                                   |
| Multilingual IPTC News Topic   | classla/multilingual-IPTC-news-topic-classifier                   |
| Financial News Sentiment       | mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis |
| DistilBERT (AG News)           | ranudee/news-category-classifier                                  |
| ValuRank Topic Classifier      | valurank/distilroberta-topic-classification                       |
| BERT (40 News Categories)      | cssupport/bert-news-class                                         |
| Custom Model                   | (User-provided HF ID)                                             |

- Click Run
- Explore
  1. Model-wise predicted labels and confidence scores
  2. Classification probability bar charts
  3. CSV download buttons
  4. Comparison tabs for each classifier

- Exporting Results
    1. After inference, you can:
       - Download per-model CSVs with results (NER entities or classification labels)
       - Download a combined CSV containing all model outputs with a model column
- Visualization Examples
  
| Type                             | Description                                             |
| -------------------------------- | ------------------------------------------------------- |
| **Entity Highlights**            | Colored spans for each recognized entity                |
| **Token Importance Bar**         | Proxy attention plot showing token relevance            |
| **Classification Probabilities** | Horizontal bar chart of predicted labels and scores     |
| **Model Tabs**                   | Side-by-side comparison of results from multiple models |

 
---
       
## Example Input Text
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California in April 1976.
The company’s headquarters, known as Apple Park, is located in Cupertino, part of the San Francisco Bay Area.
Tim Cook became the CEO of Apple after Steve Jobs resigned in August 2011.
In November 2023, Apple partnered with the University of California, Berkeley to research advanced AI systems.

---

## Future Enhancements
- Add multilingual NER models (e.g., mBERT, XLM-RoBERTa large multilingual)
- Integrate zero-shot and multi-label classification
- Integrate sentence segmentation for long documents
- Add performance benchmarking across models
- Add per-entity confidence visualizations
