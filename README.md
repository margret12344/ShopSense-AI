
# ShopSense AI

An NLP-based E-commerce Product Classification System.

## Overview

ShopSense AI automatically classifies e-commerce product descriptions into four categories:

- Electronics
- Household
- Books
- Clothing & Accessories

The project uses TF-IDF for text feature extraction and compares multiple Machine Learning algorithms to select the best-performing model using Cross-Validation Macro F1.

## Machine Learning Models

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Classifier

## Project Structure

```text
ShopSense-AI/
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
├── models/
├── reports/
├── src/
└── assets/
```

## Technologies

- Python
- Pandas
- Scikit-learn
- NLP
- TF-IDF
- Streamlit
- Joblib

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ShopSense-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Application Features

- Interactive dashboard
- Single product classification
- Batch CSV classification
- Downloadable prediction results
- Model comparison
- Cross-validation analysis
- Misclassified examples analysis

## Model Artifacts

The application loads the trained model and metadata from the `models/` directory.

Evaluation reports are stored in the `reports/` directory.

## Author

Margret Atef Shawky
