<div align="center">

# EmotiCare

**AI-Powered Mental Health Assistant**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![BERT](https://img.shields.io/badge/BERT-Contextual%20NLP-8B5CF6?style=flat-square)](https://arxiv.org/abs/1810.04805)

*Empathy at scale — BERT-based emotion detection and crisis support for mental health applications.*

</div>

---

## What is EmotiCare?

EmotiCare is an NLP system that detects emotional states and crisis signals in user text, then generates contextually empathetic responses. It addresses a real problem: most chatbots respond to *what* someone says, not *how* they feel. EmotiCare reads the emotional subtext.

**Use cases:** Mental health platforms · Crisis hotline support tools · Empathetic customer support · Wellness applications

---

## How it works

```
User input text
      │
      ▼
┌─────────────────────────┐
│  BERT Emotion Encoder   │  Fine-tuned on GoEmotions dataset
│  (bert-base-uncased)    │  28 emotion classes → 7 primary
└──────────┬──────────────┘
           │  emotion vector + confidence scores
           ▼
┌─────────────────────────┐
│  Crisis Detection Layer │  Binary classifier: crisis / non-crisis
│  (threshold: 0.72)      │  Triggers escalation protocol if positive
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Empathetic Response    │  Template + generative response blending
│  Generator              │  Tone matched to detected emotion
└─────────────────────────┘
           │
           ▼
     Empathetic reply
  + resource suggestions (if crisis)
```

---

## Model architecture

| Component | Model | Dataset | Task |
|---|---|---|---|
| Emotion encoder | `bert-base-uncased` fine-tuned | GoEmotions (58k samples) | 28-class emotion classification |
| Crisis detector | Logistic classifier on BERT embeddings | Custom crisis corpus | Binary: crisis / non-crisis |
| Response generator | Template + GPT-2 blending | Empathetic Dialogues | Contextual empathetic reply |

**Emotion classes detected:**
`admiration` `amusement` `anger` `annoyance` `approval` `caring` `confusion` `curiosity` `desire` `disappointment` `disapproval` `disgust` `embarrassment` `excitement` `fear` `gratitude` `grief` `joy` `love` `nervousness` `optimism` `pride` `realization` `relief` `remorse` `sadness` `surprise` `neutral`

---

## Key results

| Metric | Score |
|---|---|
| Emotion classification accuracy | 87.3% |
| Crisis detection precision | 91.2% |
| Crisis detection recall | 89.6% |
| Response empathy rating (human eval) | 4.1 / 5.0 |
| Avg inference latency | < 120ms |

> High recall on crisis detection is the priority metric — missing a crisis signal is a far worse error than a false positive.

---

## Tech stack

```
NLP Pipeline
├── Tokenization      HuggingFace Tokenizers (WordPiece)
├── Encoder           BERT (bert-base-uncased, 110M params)
├── Fine-tuning       PyTorch + HuggingFace Trainer API
├── Emotion labels    GoEmotions (Google Research)
└── Inference         ONNX export for production speed

Infrastructure
├── Training          Google Colab (GPU: T4)
├── Experiment track  Weights & Biases
├── Data pipeline     Pandas + scikit-learn preprocessing
└── Evaluation        Classification report + confusion matrix
```

---

## Project structure

```
EmotiCare/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # GoEmotions EDA
│   ├── 02_bert_finetuning.ipynb       # Fine-tuning pipeline
│   ├── 03_crisis_detection.ipynb      # Crisis classifier training
│   └── 04_response_generation.ipynb   # Response blending
├── src/
│   ├── emotion_encoder.py             # BERT fine-tune wrapper
│   ├── crisis_detector.py             # Binary crisis classifier
│   └── response_generator.py         # Empathetic response engine
├── data/
│   └── crisis_corpus/                 # Curated crisis signal examples
├── models/                            # Saved checkpoints
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/ArchanaChetan07/EmotiCare_-AI-Powered-Mental-Health-Assistant
cd EmotiCare_-AI-Powered-Mental-Health-Assistant
pip install -r requirements.txt

# Run the full pipeline
jupyter notebook notebooks/02_bert_finetuning.ipynb
```

---

## Sample output

```
Input:  "I've been feeling really hopeless lately. Nothing seems worth it anymore."

Detected emotion:  grief (0.84 confidence)
Crisis signal:     YES (0.91 confidence) ⚠️

Response:
  "It sounds like you're carrying something really heavy right now.
   Feeling hopeless is incredibly hard, and what you're experiencing matters.
   You don't have to face this alone — would you like to talk more about
   what's been going on? If you're in crisis, please reach out to the
   988 Suicide & Crisis Lifeline (call or text 988)."
```

---

## Research context

This project was developed as part of research into **human-centered machine intelligence** — building AI systems that recognize and respond to emotional context, not just semantic content. The EmotiCare framework is designed to be integrated into existing mental health platforms as an empathy layer, not a replacement for professional care.

---

## Author

**Archana Suresh Patil** — NLP & AI Research  
MS Data Science · University of San Diego · GPA 3.9  
📬 apatil@sandiego.edu · [LinkedIn](https://linkedin.com/in/archana-suresh-patil-792213245) · [GitHub](https://github.com/ArchanaChetan07)
