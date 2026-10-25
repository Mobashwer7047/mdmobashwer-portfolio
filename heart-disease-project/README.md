# HeartWise — Heart Disease Prediction Using Machine Learning

A Streamlit portfolio application demonstrating dataset exploration, preprocessing pipelines, model comparison, evaluation metrics, and a prediction interface.

## Important dataset note
The built-in offline dataset is **scikit-learn's breast-cancer dataset**, included only so the app runs without downloading a file. It is **not a heart-disease dataset**. For valid heart-disease analysis, upload a documented heart-disease CSV and verify its feature meanings and target coding.

## Features
- Built-in demo dataset and user CSV upload
- Dataset preview, missing-value summary, class counts, distributions, correlation heatmap
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and KNN
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve
- Interactive model prediction demo
- CSV download
- Medical-use warning

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. Open Streamlit Community Cloud and create an app from that repository.
3. Set the main file path to `app.py`.

## CSV upload format
- Include one binary target column named `target`, `num`, `heart_disease`, `disease`, `output`, or `label`.
- Target values should be `0` and `1` (`1` is the positive class).
- Numeric feature columns are used for modelling.
- Do not upload identifiable patient information.

## Responsible use
This project is for education and portfolio demonstration only. It is not a medical device and must not be used to diagnose, treat, or rule out disease. A qualified clinician should interpret health concerns.
