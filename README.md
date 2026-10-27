# IMDb Movie Review Sentiment Analysis

This project compares traditional machine learning and transformer-based
models for classifying IMDb movie reviews as positive or negative. It was
completed as part of the Advanced Machine Learning course at Malmö University.

## Project Overview

We compared three approaches:

- **TF-IDF and Logistic Regression** as a traditional machine learning baseline.
- **DistilBERT**, a lightweight transformer model fine-tuned for sentiment analysis.
- **RoBERTa**, a transformer model fine-tuned for the same task.

We also explored DistilBERT attention maps to visualize attention patterns
in movie reviews.

## Dataset

The project uses the IMDb movie review dataset, containing 50,000 English
reviews: 25,000 positive and 25,000 negative.

The data was split into:

- 40,000 training reviews
- 5,000 validation reviews
- 5,000 test reviews

## Results

| Model | Accuracy |
| --- | ---: |
| Logistic Regression + TF-IDF | 89.28% |
| DistilBERT | 92.30% |
| RoBERTa | 91.28% |

DistilBERT achieved the highest accuracy in our experiments. RoBERTa required
more computational resources, and GPU memory and runtime limitations
restricted our experiments and hyperparameter tuning.

We also evaluated the models using precision, recall, F1-score and confusion
matrices. See the project report for more details.

## Tools and Technologies

- Python
- Google Colab
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Pandas and NumPy

## Repository Contents

- **Logistic Regression and DistilBERT notebook:** preprocessing, training,
  evaluation and attention visualization.
- **RoBERTa notebook:** fine-tuning and evaluation.
- **Project report:** methodology, results and discussion.

## Running the Notebooks

Open the notebooks in Google Colab to explore the code. Follow the steps and
data-loading instructions in each notebook. Transformer training may require
a GPU and can take considerable time.

## Limitations

The project focuses on English movie reviews and binary sentiment
classification, so the results may not generalize to other languages or
domains. Attention maps help visualize attention patterns, but they do not
provide a complete explanation of model decisions.

## Authors

Molu Sunder and Maryam Asghar  
Malmö University
