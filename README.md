# Fake News Detection - NLP + ML Project

## Project Overview

This project tackles the challenge of detecting fake news using Natural Language Processing (NLP) and classic Machine Learning (ML) algorithms. A real-world problem with massive implications in media and politics, fake news detection is a strong use case for text classification. This app allows users to enter a news headline and content, and predicts whether the news is likely **Fake** or **Real**.

## Dataset Description

Used two open-source CSV files from Kaggle:

- **True.csv**: Verified real news articles
- **Fake.csv**: Fabricated or misleading articles

Each dataset contains news `title`, `text`, and other metadata. We combined and labeled them (`1 = REAL`, `0 = FAKE`) before processing.

## Features & Models

The project includes the following components:

### Preprocessing:

- Text cleaning: Lowercasing, removing punctuation, links, etc.
- TF-IDF vectorization (1000 features)
- Class balancing via downsampling

### ML & Deep Learning Models Implemented:

- Logistic Regression
- Random Forest
- Basic Neural Network (TF-IDF based)
- Long Short-Term Memory (LSTM)
- XGBoost (used for deployment)

### Evaluation:

- Accuracy, precision, recall, F1-score
- Classification report for each model
- Confusion matrix visualizations for detailed error analysis

## Model Performance Comparison

| Model                | Test Accuracy | Notes                                                  |
| -------------------- | ------------- | ------------------------------------------------------ |
| Logistic Regression  | 98.0%         | Strong baseline, high precision and recall             |
| Random Forest        | 100.0%        | Near-perfect accuracy, potential overfitting           |
| Basic Neural Network | 98.8%         | Generalized well, fast to train                        |
| LSTM                 | 87.5%         | Captures sequence well, but lower precision on class 0 |
| XGBoost (Deployed)   | 100.0%        | Excellent accuracy, robust performance                 |

## Visualizations

- [Confusion Matrices and Classification Reports](Fake_News_Detector.ipynb)\
  This section provides visual diagnostics and metrics for each model, helping to analyze prediction quality and model behavior beyond accuracy alone.

## Streamlit App Deployment

Created a working `app_final.py` that:

- Loads the TF-IDF vectorizer and XGBoost model
- Accepts user input for title + content
- Predicts whether the news is FAKE or REAL

## Known Limitation

Due to repeated issues with `.pkl` corruption during model saving/loading, the Streamlit app **may show inaccurate predictions** despite correct logic. The model sometimes outputs "REAL" regardless of the input.

> This highlights a very real industry challenge: model alignment and deployment are **not trivial**. Kept the buggy app live to show authentic effort, but clearly note this limitation.

## Future Improvements

- Switch to BERT or further tune the LSTM for better generalization
- Implement SMOTE or class weighting instead of downsampling
- Improve app accuracy with validation hooks
- Dockerize the app for clean deployment

## How to Run Locally

1. Clone the repo
2. Run the Jupyter notebook to train + save the model:
   - `Fake_News_Detector.ipynb`
3. Launch the app:

```bash
pip install -r requirements.txt
streamlit run app_final_corrected.py
```

## What This Project Demonstrates

- Text data preprocessing and pipeline building
- Multi-model and neural network implementation for NLP classification
- Evaluation with classification reports and confusion matrices
- Real-world debugging (deployment issues, prediction mismatch)
- Persistence and clarity in documenting challenges

## Credits

Created by Junghyun Cheon. Inspired by fake news detection competitions and tutorials on Kaggle and Medium.

---

**If you're a recruiter or hiring manager**: I welcome any feedback and would love to discuss this project in more depth!

