# Fake News Detection using Semantic Classification

## Objective

The goal of this project is to build a **semantic classification model** for detecting fake news articles. The approach focuses on understanding the **meaning** behind the text using **Word2Vec embeddings**, rather than relying solely on syntactic patterns. The project provides hands-on experience in applying **Natural Language Processing (NLP)** techniques to real-world problems and training supervised models to make data-driven decisions.

## Business Context

The exponential rise of misinformation and fake news in digital media has made it challenging to trust online content. With thousands of articles published every day, there's a growing need for automated systems that can **classify news articles as "True" or "Fake"**, helping combat misinformation and safeguard public awareness.

This project aims to tackle this problem by building a **machine learning pipeline** that leverages **semantic relationships** between words. Instead of keyword matching or superficial patterns, the focus is on learning **meaningful word representations** and **contextual relationships** to improve the reliability of fake news detection.

## Pipeline Overview

The project follows a structured pipeline from data ingestion to model evaluation:

### 1. Data Preparation

* Load and merge news datasets (true and fake).
* Assign binary labels (`1` for true, `0` for fake).

### 2. Text Preprocessing

* Clean raw text by removing punctuation, stopwords, and special characters.
* Extract **nouns** using POS tagging and apply **lemmatization** for semantic clarity.

### 3. Train-Validation Split

* Split the dataset into **70% training** and **30% validation** for fair evaluation.

### 4. Exploratory Data Analysis (EDA)

* Visualize character length distributions of processed text.
* Generate word clouds to identify frequent terms in true vs. fake news.
* Analyze top **unigrams**, **bigrams**, and **trigrams** separately for both classes.

### 5. Feature Extraction

* Generate **Word2Vec embeddings** using pre-trained `GoogleNews-vectors-negative300`.
* Average word vectors for each article to get document-level embeddings.

### 6. Model Training and Evaluation

Train and evaluate the following models on the embedded text data:

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**

Performance is measured using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

### 7. Summary of Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.9013   | 0.8911    | 0.9033 | 0.8972   |
| Decision Tree       | 0.8851   | 0.8724    | 0.8972 | 0.8846   |
| Random Forest       | 0.9057   | 0.9048    | 0.8967 | 0.9007   |

**Note:** Random Forest delivered the best performance with an F1-score of **0.9007**, balancing both precision and recall effectively.

---

## Conclusion

This project successfully demonstrates that **semantic-aware classification** can outperform traditional syntactic methods in detecting fake news. By using **noun-focused lemmatization** and **Word2Vec embeddings**, the model captures the deeper context of the news articles.

Among the three models tested, **Random Forest** emerged as the top performer, achieving **90.6% accuracy** and an **F1-score of 0.901**. This shows the potential of combining **POS-tagged lemmatization** and **distributed word representations** in real-world NLP pipelines.


