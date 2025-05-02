# LLM‑Based Sentiment Analysis on Tesla

**Course**: INFO‑6153 Natural Language Processing 2  
**Student**: **Ahkar Htet** (ID 1230551)  
**Project**: Fine‑Tuned LLaMA‑2 (7 B) for Continuous Sentiment Scoring  
**Date issued**: 2025‑04‑12

---

## Abstract
This project fine‑tunes the 7‑billion‑parameter **LLaMA‑2** language model on a Twitter/News corpus about **Tesla, Inc.** to predict *continuous* sentiment scores (0–100).  A lightweight **LoRA** adapter, trained in 4‑bit precision with **bitsandbytes**, enables accurate inference within Google Colab’s memory limits.  Evaluation on live NewsAPI articles shows an average test loss ≈ **1.03**, confirming good generalisation. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

---

## 1  Introduction
Public opinion on Tesla has a measurable impact on its stock price and brand perception. Traditional classification (positive vs negative) misses nuance; therefore we predict a **fine‑grained score** that reflects sentiment intensity. Large Language Models (LLMs) such as LLaMA‑2 provide strong zero‑shot capability, but domain fine‑tuning further boosts accuracy while keeping compute costs manageable via parameter‑efficient methods. :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

---

## 2  Dataset Preparation
| Source | Records | Notes |
|--------|---------|-------|
| Kaggle “Twitter Dataset – Tesla” | 10 016 raw tweets | Cleaned → 10 k; limited to 200 for prototyping :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5} |
| NewsAPI live articles | 10 latest | Used exclusively for final test set :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7} |

Cleaning removes URLs, mentions, hashtags, emojis, and punctuation, then converts to lowercase. Continuous labels are generated with **DistilBERT** (`distilbert‑base‑uncased‑sst‑2`) and scaled to 0–100. The final split is 80 % train / 20 % validation. :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}

---

## 3  Model & Fine‑Tuning Strategy

### 3.1 Base model
`meta‑llama/Llama‑2‑7b‑hf`, loaded in **4‑bit NF4** quantisation to fit a single Colab GPU. :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}

### 3.2 LoRA configuration
```text
rank (r)        : 8
alpha           : 16
target modules  : ["q_proj", "v_proj"]
dropout         : 0.05
trainable params: 7.4 M (≈0.1 % of full model)
