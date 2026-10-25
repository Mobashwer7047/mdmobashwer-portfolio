import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

st.set_page_config(
    page_title="HeartWise | ML Health Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1400px;}
.hero {padding: 1.25rem 1.4rem; border-radius: 18px; background: linear-gradient(120deg,#182235,#263c57); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.15rem;}
.hero p {margin: .45rem 0 0; color: #d9e5f3;}
.note {padding: .8rem 1rem; border-left: 4px solid #e0a33a; background: rgba(224,163,58,.10); border-radius: 8px;}
.small-muted {font-size: .9rem; opacity: .8;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); padding: .8rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
FEATURE_LABELS = {
    "age": "Age (years)", "sex": "Sex (dataset code: 1=male, 0=female)",
    "cp": "Chest pain type (0–3)", "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Cholesterol (mg/dL)", "fbs": "Fasting blood sugar >120 mg/dL (1=yes)",
    "restecg": "Resting ECG (0–2)", "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina (1=yes)", "oldpeak": "ST depression",
    "slope": "ST segment slope (0–2)", "ca": "Major vessels (0–4)", "thal": "Thalassemia code (0–3)"
}
TARGET_CANDIDATES = ["target", "num", "heart_disease", "disease", "output", "label"]

@st.cache_data
def load_demo_data():
    # Use sklearn's built-in dataset as a guaranteed, offline demo source.
    # It is deliberately labelled as a proxy demonstration dataset, NOT heart-disease data.
    raw = load_breast_cancer(as_frame=True)
    frame = raw.frame.copy()
    frame = frame.rename(columns={c: c.replace(" ", "_").replace("(", "").replace(")", "") for c in frame.columns})
    frame["target"] = raw.target.astype(int)
    frame["target"] = frame["target"].map({0: 1, 1: 0})
    return frame

@st.cache_resource
def train_models(data_hash, X, y):
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    fitted, rows = {}, []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe[-1], "predict_proba") else pred
        try:
            auc = roc_auc_score(y_test, score)
        except Exception:
            auc = float("nan")
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc,
        })
        fitted[name] = pipe
    return fitted, pd.DataFrame(rows).sort_values("F1-score", ascending=False), X_train, X_test, y_train, y_test

def normalize_target(series):
    if pd.api.types.is_numeric_dtype(series):
        vals = sorted(pd.Series(series.dropna().unique()).tolist())
        if len(vals) == 2:
            # Convention: 1 means disease/positive; 0 means no disease.
            return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map({
        "yes": 1, "disease": 1, "positive": 1, "present": 1, "true": 1,
        "no": 0, "no disease": 0, "negative": 0, "absent": 0, "false": 0
    })
    return mapped

def prepare_upload(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        return None, None, None, "Target column not found. Name it target, num, heart_disease, disease, output, or label."
    y = normalize_target(df[target_col])
    if y.isna().any():
        return None, None, None, "Target must contain two classes encoded as 0/1 or common labels such as yes/no."
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=np.number)
    if X.shape[1] < 2:
        return None, None, None, "At least two numeric feature columns are required."
    keep = y.notna()
    X, y = X.loc[keep], y.loc[keep].astype(int)
    if y.nunique() != 2:
        return None, None, None, "The target must have exactly two classes."
    if y.value_counts().min() < 2:
        return None, None, None, "Each target class needs at least two rows for a train/test split."
    return X, y, target_col, None

st.markdown("""
<div class="hero">
  <h1>❤️ HeartWise — Heart Disease ML Lab</h1>
  <p>Interactive machine-learning analytics, model comparison, and an educational prediction demo.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="note"><b>Important:</b> This is an educational software demonstration, not a medical device. Do not use its output to diagnose, treat, or rule out heart disease. Seek a qualified clinician for medical concerns.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Workspace")
    page = st.radio("Navigate", [
        "Overview", "Explore Dataset", "Model Lab", "Prediction Demo", "About & Data Guide"
    ])
    st.divider()
    st.caption("Project: Heart Disease Prediction Using Machine Learning")
    st.caption("Student portfolio demo • Python + Streamlit")

st.subheader("Choose your dataset")
source = st.radio("Data source", ["Built-in demonstration dataset", "Upload my CSV"], horizontal=True)
uploaded = None
if source == "Upload my CSV":
    uploaded = st.file_uploader("Upload CSV (must include a binary target column)", type=["csv"])
    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded)
            X, y, target_name, err = prepare_upload(source_df)
            if err:
                st.error(err)
                st.stop()
            data = X.copy()
            data["target"] = y
            st.success(f"Loaded {len(data)} rows and {X.shape[1]} numeric features. Target column: {target_name}")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
    else:
        st.info("Upload a CSV to continue, or choose the built-in demonstration dataset.")
        st.stop()
else:
    data = load_demo_data()
    st.warning("The built-in offline dataset is a breast-cancer classification dataset used only to demonstrate the app workflow. It is NOT a heart-disease dataset. For valid heart-disease modelling, upload a properly documented heart-disease dataset.")

target = "target"
X = data.drop(columns=[target]).select_dtypes(include=np.number)
y = data[target].astype(int)
if y.nunique() != 2 or y.value_counts().min() < 2:
    st.error("This app needs a binary target with at least two rows in each class.")
    st.stop()

data_hash = str(pd.util.hash_pandas_object(data, index=True).sum())
models, scores, X_train, X_test, y_train, y_test = train_models(data_hash, X, y)

if page == "Overview":
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", f"{len(data):,}")
    c2.metric("Numeric features", f"{X.shape[1]}")
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    c4.metric("Models compared", f"{len(models)}")
    st.markdown("### Model leaderboard")
    st.dataframe(scores.style.format({c: "{:.3f}" for c in scores.columns if c != "Model"}), use_container_width=True, hide_index=True)
    best = scores.iloc[0]
    st.info(f"Top model by F1-score on this held-out split: **{best['Model']}** (F1 = {best['F1-score']:.3f}). This is a dataset-specific benchmark, not a clinical performance guarantee.")
    st.markdown("### Workflow")
    st.write("Dataset → data checks → stratified train/test split → preprocessing pipeline → model comparison → demo prediction.")
elif page == "Explore Dataset":
    st.subheader("Dataset Explorer")
    a, b = st.columns(2)
    with a:
        st.metric("Rows", len(data))
        st.dataframe(data.head(15), use_container_width=True)
    with b:
        st.write("Missing values by column")
        st.dataframe(data.isna().sum().rename("Missing").to_frame(), use_container_width=True)
        st.write("Target class counts")
        st.dataframe(y.value_counts().rename_axis("Target class").rename("Rows").to_frame(), use_container_width=True)
    st.markdown("### Numeric feature distributions")
    numeric_cols = list(X.columns)
    chosen = st.selectbox("Choose a feature", numeric_cols)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(data=data, x=chosen, hue="target", kde=True, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(f"{chosen} distribution by target class")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("### Correlation heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(data.select_dtypes(include=np.number).corr(), cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("Download current dataset CSV", data.to_csv(index=False).encode("utf-8"), "heartwise_dataset.csv", "text/csv")
elif page == "Model Lab":
    st.subheader("Model Evaluation Lab")
    st.caption("Scores are calculated on a held-out test split. Performance depends on dataset quality and representativeness.")
    st.dataframe(scores.style.format({c: "{:.3f}" for c in scores.columns if c != "Model"}), use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect model", list(models.keys()))
    pipe = models[selected]
    pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    left, right = st.columns(2)
    with left:
        st.write("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); fig.tight_layout(); st.pyplot(fig)
    with right:
        st.write("Classification report")
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.write("ROC curve")
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
        ax.plot([0, 1], [0, 1], "--", label="Chance")
        ax.legend(); fig.tight_layout(); st.pyplot(fig)
    except Exception:
        st.info("ROC curve is not available for this uploaded dataset/model.")
elif page == "Prediction Demo":
    st.subheader("Prediction Demo")
    st.write("Enter values matching the selected dataset's feature columns. These outputs demonstrate model behaviour only.")
    model_name = st.selectbox("Prediction model", list(models.keys()))
    chosen_model = models[model_name]
    values = {}
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        series = X[col].dropna()
        default = float(series.median()) if not series.empty else 0.0
        lo = float(series.min()) if not series.empty else -1000000.0
        hi = float(series.max()) if not series.empty else 1000000.0
        with cols[i % 2]:
            values[col] = st.number_input(
                f"{col}", min_value=lo, max_value=hi, value=default,
                help=f"Observed dataset range: {lo:.3g} to {hi:.3g}"
            )
    if st.button("Run educational prediction", type="primary"):
        sample = pd.DataFrame([values], columns=X.columns)
        label = int(chosen_model.predict(sample)[0])
        st.markdown("### Model output")
        st.metric("Predicted class", str(label))
        if hasattr(chosen_model[-1], "predict_proba"):
            prob = float(chosen_model.predict_proba(sample)[0, 1])
            st.metric("Model score for class 1", f"{prob:.1%}")
        st.warning("This is a model classification output, not a personal medical risk score or diagnosis. The built-in dataset is not heart-disease data.")
elif page == "About & Data Guide":
    st.subheader("About this project")
    st.write("HeartWise is a portfolio-style ML dashboard demonstrating dataset inspection, preprocessing pipelines, model comparison, evaluation charts, and an interactive prediction interface.")
    st.markdown("### CSV format requirements")
    st.markdown("""
- One row per example.
- Numeric feature columns are used by the models.
- A binary target column named `target`, `num`, `heart_disease`, `disease`, `output`, or `label`.
- Target convention: `1` = positive class and `0` = negative class; confirm the meaning in your dataset documentation.
- Do not upload identifiable patient information.
""")
    st.markdown("### Suggested documented heart-disease dataset")
    st.write("For a real heart-disease education project, use a reputable, documented dataset such as the UCI Heart Disease dataset and follow its license and feature definitions. Check missing values and class labels before training.")
    st.markdown("### Tech stack")
    st.write("Python • Streamlit • Pandas • NumPy • scikit-learn • Matplotlib • Seaborn")
    st.markdown("### Responsible-use notes")
    st.write("A high benchmark score does not prove clinical usefulness. Data leakage, selection bias, class imbalance, and differences between populations can mislead evaluation. This app is not for diagnosis or treatment decisions.")

st.divider()
st.caption("HeartWise ML Lab • Educational demonstration • Not for clinical use")
