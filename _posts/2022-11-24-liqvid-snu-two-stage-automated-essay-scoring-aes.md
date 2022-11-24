---
layout: post
published: true
title: 'LiqVid <> SNU:  Two Stage Automated Essay Scoring (AES)'
subtitle: Automated Essay Scoring (AES) project - LiqVid<>SNU collaboration.
date: '2022-02-18'
tags:
  - AES
  - NLP
  - Deep Language Model
  - BERT
  - Industry Collaboration
---

## Experiments
### Python Notebooks

The jupyter notebooks are described as follows:

#### AES_MLP_BERT (9/10/2022)

We compare results from 1D-CNN, 2D-CNN, and MLP training models with TF hub BERT APIs.

#### Baseline-w-CSE (7/17/2022)

Baseline training models for different experiments for the first stage (Semantic/Coherence/Prompt-relevance)  with the spelling errors corrected from the essays to improve BERT and training model performance. Experiments are as follows:

- **Semantic Model** 
	- with 2nd Last Hidden State (2LHS) BERT Embeddedings
    - with sum of Last 4 Hidden States (L4HS) BERT Embeddings
    - L4HS + Sentence Average BERT Embeddings
    - L4HS + Pooled BERT Embeddings

- **Coherence Model**
	- with 2nd Last Hidden State (2LHS) BERT Embeddedings
    - with sum of Last 4 Hidden States (L4HS) BERT Embeddings
    - L4HS + Sentence Average BERT Embeddings
    - L4HS + Pooled BERT Embeddings
    
 - **Coherence Model - Next Sentence Prediction (NSP):** Using the output from BERT model fine-tuned for next sentence prediction to evaluate local and global average sentence coherency. We perform a direct evaluation of the two models on an unseen IELTS dataset.
 
 - **Prompt Relevance Model** 
	- with 2nd Last Hidden State (2LHS) BERT Embeddedings
    - with sum of Last 4 Hidden States (L4HS) BERT Embeddings
    - L4HS + Sentence Average BERT Embeddings
    - L4HS + Pooled BERT Embeddings
    
- **Prompt Relevance Model - Cosine Similarity (COSIM):** We explore evaluating the prompt relevance using cosine similarity of pooled essay embedding and the prompt embedding. We further perform a direct evaluation of the two models on an unseen IELTS dataset. 

 
#### AES-baseline (7/17/2022)

Baseline training models for different experiments for the first stage (Semantic/Coherence/Prompt-relevance)  Experiments are as follows:

- **Semantic Model** 
	- with 2nd Last Hidden State (2LHS) BERT Embeddedings
    - with sum of Last 4 Hidden States (L4HS) BERT Embeddings
    - L4HS + Sentence Average BERT Embeddings
    - L4HS + Pooled BERT Embeddings

- **Coherence Model **
	- with 2nd Last Hidden State (2LHS) BERT Embeddedings
    - with sum of Last 4 Hidden States (L4HS) BERT Embeddings
    - L4HS + Sentence Average BERT Embeddings
    - L4HS + Pooled BERT Embeddings
 
 - **Prompt Relevance Model** 
	- with 2nd Last Hidden State (2LHS) BERT Embeddedings
    - with sum of Last 4 Hidden States (L4HS) BERT Embeddings
    - L4HS + Sentence Average BERT Embeddings
    - L4HS + Pooled BERT Embeddings
    
**2nd Stage Model Selection:** We explore different tree-based models (including XGBoost and RF) for 2nd stage regression model selection.

## GitHub Repository 
[https://github.com/agastyaseth/aes-two-stage](https://github.com/agastyaseth/aes-two-stage)

## Preprint Article
[https://github.com/agastyaseth/aes-two-stage/blob/main/AES-Paper-Feb2022-v1.0.pdf](https://github.com/agastyaseth/aes-two-stage/blob/main/AES-Paper-Feb2022-v1.0.pdf)


