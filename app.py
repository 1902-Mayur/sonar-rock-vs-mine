from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sonar Rock vs Mine | Deep Ocean Intelligence", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "sonar_pipeline.joblib"
CSS_PATH = BASE_DIR / "style.css"
DATA_CANDIDATES = [
    BASE_DIR / "sonar_dataset(3).csv",
    BASE_DIR / "sonar_dataset(2).csv",
    BASE_DIR / "sonar_dataset(1).csv",
    BASE_DIR / "sonar_dataset.csv",
    BASE_DIR / "sonar.csv",
]

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            data = pd.read_csv(path)
            # The supplied dataset has a header row of feature indexes 0..59 and target label 60.
            if str(data.columns[-1]).strip() == "60" and data.shape[1] == 61:
                data.columns = [f"band_{i:02d}" for i in range(60)] + ["label"]
            elif data.shape[1] == 61:
                data.columns = [f"band_{i:02d}" for i in range(60)] + ["label"]
            if "label" not in data.columns:
                data = data.rename(columns={data.columns[-1]: "label"})
            data["label"] = data["label"].astype(str).str.strip().str.upper()
            data = data[data["label"].isin(["M", "R"])].copy()
            for col in data.columns[:-1]:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            return data, path.name
    return None, None

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

data, data_name = load_dataset()
model = load_model()
FEATURES = int(getattr(model, "n_features_in_", 60)) if model is not None else 60
CLASSES = list(getattr(model, "classes_", ["M", "R"])) if model is not None else ["M", "R"]

def mine_probabilities(probabilities):
    classes = list(model.classes_)
    return float(probabilities[classes.index("M")])

def parse_values(text):
    # Comma-separated is the preferred format; whitespace and semicolons also accepted.
    tokens = [x for x in re.split(r"[\s,;]+", text.strip()) if x]
    return [float(x) for x in tokens]

def make_feature_frame(values):
    """Build input with the exact column names used when the saved model was fitted."""
    if model is not None and hasattr(model, "feature_names_in_"):
        columns = list(model.feature_names_in_)
    else:
        columns = [f"band_{i:02d}" for i in range(FEATURES)]
    return pd.DataFrame([np.asarray(values, dtype=float)], columns=columns)

def run_prediction(values):
    if model is None:
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Put sonar_pipeline.joblib inside the artifacts folder.")
    if len(values) != FEATURES:
        raise ValueError(f"Expected exactly {FEATURES} feature values; received {len(values)}.")
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Values must be finite numbers.")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError("Every sonar feature must be between 0 and 1.")
    frame = make_feature_frame(arr)
    prediction = model.predict(frame)[0]
    probabilities = model.predict_proba(frame)[0]
    return prediction, probabilities

def radar_markup(prediction=None, confidence=None):
    # Marker is positioned inside the radar; color communicates the predicted class.
    if prediction in ("M", "R"):
        color = "#ff536b" if prediction == "M" else "#27e0b5"
        label = "MINE" if prediction == "M" else "ROCK"
        # Place the marker in the upper-right quadrant; its distance varies with confidence.
        radius = 22 + (float(confidence or 0.5) * 24)
        left = 50 + radius * 0.707
        top = 50 - radius * 0.707
        marker = (
            f'<span class="radar-result-marker" style="--marker-color:{color};left:{left:.1f}%;top:{top:.1f}%;">'
            f'<span class="radar-result-dot"></span><span class="radar-result-label">{label}</span></span>'
        )
        result_class = " has-result"
    else:
        marker = ""
        result_class = ""
    st.markdown(
        f'<div class="radar-wrap prediction-radar-wrap"><div class="radar{result_class}">'
        '<div class="radar-crosshair"></div><div class="radar-crosshair vertical"></div>'
        f'{marker}</div></div>',
        unsafe_allow_html=True,
    )

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def page_header(title, subtitle):
    st.markdown(f'<div class="page-heading"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

# Sidebar / navigation
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-title">🌊 Sonar Rock vs Mine</div><div class="brand-subtitle">Underwater object classification with machine learning</div></div>', unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Home", "Project Overview", "Data Analysis", "Visualizations", "Model Training", "Prediction System", "About Developer"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    if model is not None:
        st.success("Trained model loaded")
        st.caption(f"Model input: {FEATURES} bands")
        st.caption(f"Classes: {', '.join(CLASSES)}")
    else:
        st.warning("Model artifact not found")
        st.caption("Expected: artifacts/sonar_pipeline.joblib")
    st.markdown("---")
    st.markdown("**Evaluation note**")
    st.caption("Hold-out results are from the notebook's tuned Extra Trees evaluation. This is an educational demo, not an operational mine detector.")

if page == "Home":
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="hero-top-row"><div class="eyebrow">CAPSTONE · MACHINE LEARNING</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Is that echo a rock,<br>or a mine?</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-description">A sonar pulse returns 60 energy-band measurements. This project explores the data, compares classifiers, tunes a model and provides an interactive Rock/Mine prediction demo.</div>', unsafe_allow_html=True)
        st.markdown('<div class="pill-row"><span class="pill">208 labelled sonar returns</span><span class="pill">60 numeric bands</span><span class="pill">17 classifiers explored</span><span class="pill">scikit-learn · Extra Trees</span></div>', unsafe_allow_html=True)
    with right:
        radar_markup()
    st.markdown("")
    cols = st.columns(4)
    values = [
        ("Hold-out accuracy", "90.48%", "Tuned Extra Trees · 42 test rows"),
        ("Mine recall", "100.0%", "22 of 22 mine samples found"),
        ("Mine F1-score", "91.67%", "Hold-out test set"),
        ("5-fold CV accuracy", "83.74%", "Mean · standard deviation 6.21 pp"),
    ]
    for col, (label, value, note) in zip(cols, values):
        with col: metric_card(label, value, note)
    st.markdown('<div class="section-title">Explore the platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="platform-grid">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="section-card"><div class="feature-icon">🎯</div><h3>Try the predictor</h3><p>Paste a sample echo or enter 60 values to get a Rock/Mine prediction and model probabilities.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card"><div class="feature-icon">📊</div><h3>Explore the data</h3><p>Inspect descriptive statistics, class balance, and the distribution of each energy band.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="section-card"><div class="feature-icon">🧠</div><h3>Review model results</h3><p>Understand hold-out metrics, tuning results, cross-validation and limitations.</p></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-note">These figures are taken from the notebook\'s tuned Extra Trees results. They differ from the LightGBM example shown in the reference screenshots.</div>', unsafe_allow_html=True)

elif page == "Project Overview":
    page_header("Project Overview", "Problem, method and an honest read of the results")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="section-card"><div class="feature-icon">⚓</div><h3>The problem</h3><p>Underwater mines and rocks can produce sonar returns that are difficult to distinguish. The dataset labels each return as M (mine) or R (rock). Missing a mine is a high-cost error, so mine recall is an important metric alongside false alarms and overall performance.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="section-card"><div class="feature-icon">📡</div><h3>How sonar data works</h3><p>Each sample contains 60 numeric energy measurements across frequency bands. The classifier learns patterns in those measurements associated with the two labels. These are processed features, not raw sonar waveforms.</p></div>', unsafe_allow_html=True)
    st.markdown("## Method")
    steps = [
        ("01", "Understand & audit", "208 labelled observations, 60 numeric features, no missing values or duplicate rows in the cleaned dataset."),
        ("02", "Exploratory analysis", "Review distributions, class balance, descriptive statistics and relationships between energy bands."),
        ("03", "Leakage-aware evaluation", "Use a stratified train/test split and stratified cross-validation on training data."),
        ("04", "Compare classifiers", "Evaluate a range of baseline and advanced classifiers using accuracy, mine precision/recall, F1 and ROC-AUC."),
        ("05", "Tune & validate", "Randomized search compared Extra Trees, Gradient Boosting and SVM candidates."),
        ("06", "Save & serve", "Save the fitted Extra Trees pipeline and use the same artifact for interactive predictions."),
    ]
    for num, title, desc in steps:
        st.markdown(f'<div class="method-row"><span class="step-num">{num}</span><div><strong>{title}</strong><p>{desc}</p></div></div>', unsafe_allow_html=True)
    st.markdown("## Results at a glance")
    cols = st.columns(4)
    for col, (lab, val, note) in zip(cols, [
        ("Selected model", "Extra Trees", "Tuned pipeline"),
        ("5-fold CV accuracy", "83.74%", "Mean on training folds"),
        ("Hold-out accuracy", "90.48%", "42 test samples"),
        ("Hold-out mine recall", "100%", "22 / 22 mine samples"),
    ]):
        with col: metric_card(lab, val, note)
    st.warning("Read this before quoting the numbers: the hold-out set contains only 42 samples, so a few predictions can materially change the percentages. Cross-validation mean accuracy (83.74%) is a useful stability estimate, not a guarantee of real-world performance.")
    st.markdown("## Future scope")
    st.write("Evaluate on independent sonar data; investigate probability calibration and abstention; assess feature importance; test raw-waveform or spectrogram deep-learning models when suitable data are available; and document domain-specific false-alarm costs.")

elif page == "Data Analysis":
    if data is None:
        st.error("Dataset CSV not found. Place sonar_dataset(2).csv beside app.py.")
    else:
        page_header("Data Analysis", "The labelled 60-band sonar dataset")
        X = data.drop(columns=["label"])
        y = data["label"]
        cols = st.columns(5)
        for col, (lab, val) in zip(cols, [
            ("Samples", f"{len(data)}"), ("Features", f"{X.shape[1]}"),
            ("Missing values", f"{int(data.isna().sum().sum())}"),
            ("Duplicate rows", f"{int(data.duplicated().sum())}"), ("Classes", f"{y.nunique()}"),
        ]):
            with col: metric_card(lab, val)
        tab_preview, tab_stats, tab_balance, tab_bands = st.tabs(["Preview", "Statistics", "Class balance", "Band explorer"])
        with tab_preview:
            st.caption(f"Loaded from {data_name}")
            st.dataframe(data.head(15), use_container_width=True, height=380)
            st.download_button("Download cleaned dataset CSV", data.to_csv(index=False).encode("utf-8"), "sonar_dataset_cleaned.csv", "text/csv")
        with tab_stats:
            st.markdown("### Descriptive statistics")
            st.dataframe(X.describe().T.rename_axis("band").round(4), use_container_width=True, height=440)
            st.markdown("### Class-wise mean values")
            class_means = data.groupby("label")[X.columns].mean().T
            st.dataframe(class_means.round(4), use_container_width=True, height=350)
        with tab_balance:
            counts = y.value_counts().reindex(["R", "M"]).dropna()
            c1, c2 = st.columns([1, 1])
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Rock (R)", "Mine (M)"], [int(counts.get("R", 0)), int(counts.get("M", 0))])
                ax.set_ylabel("Samples"); ax.set_title("Class distribution")
                ax.bar_label(ax.containers[0]); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            with c2:
                st.markdown("### Class share")
                share = (y.value_counts(normalize=True) * 100).reindex(["R", "M"]).dropna()
                st.dataframe(pd.DataFrame({"Class": ["Rock (R)", "Mine (M)"], "Count": [int(counts.get("R", 0)), int(counts.get("M", 0))], "Share (%)": [round(float(share.get("R", 0)), 1), round(float(share.get("M", 0)), 1)]}), use_container_width=True)
                st.caption("Class proportions are calculated from the loaded CSV.")
        with tab_bands:
            chosen = st.selectbox("Choose an energy band", list(X.columns), index=0)
            fig, ax = plt.subplots(figsize=(8, 4))
            for label, name in [("R", "Rock"), ("M", "Mine")]:
                vals = data.loc[data["label"] == label, chosen]
                ax.hist(vals, bins=18, alpha=.55, label=name)
            ax.set_title(f"Distribution of {chosen} by class"); ax.set_xlabel("Energy value"); ax.set_ylabel("Count"); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.markdown("### Band summary")
            st.dataframe(data.groupby("label")[chosen].agg(["count", "mean", "std", "min", "median", "max"]).round(4), use_container_width=True)

elif page == "Visualizations":
    if data is None:
        st.error("Dataset CSV not found. Place sonar_dataset(2).csv beside app.py.")
    else:
        page_header("Visualizations", "Exploratory analysis generated from the dataset")
        t1, t2, t3, t4 = st.tabs(["Data & EDA", "Preprocessing", "Models", "Explainability & final"])
        X = data.drop(columns=["label"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                counts = data["label"].value_counts().reindex(["R", "M"])
                ax.bar(["Rock (R)", "Mine (M)"], counts.values)
                ax.set_title("Class distribution"); ax.set_ylabel("Count"); ax.bar_label(ax.containers[0]); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(counts.values, labels=["Rock", "Mine"], autopct="%1.1f%%", startangle=90)
                ax.set_title("Class share"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.markdown("### Histograms of representative bands")
            selected = st.multiselect("Select bands", list(X.columns), default=list(X.columns[:4]))
            if selected:
                fig, ax = plt.subplots(figsize=(10, 4))
                for band in selected:
                    ax.hist(X[band], bins=20, alpha=.35, label=band)
                ax.set_xlabel("Energy value"); ax.set_ylabel("Count"); ax.legend(ncol=4); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.markdown("### Feature correlation")
            n = st.slider("Number of leading bands to show", 5, 30, 15)
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(X.iloc[:, :n].corr(), cmap="mako", ax=ax, square=True)
            ax.set_title(f"Correlation heatmap: first {n} bands"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with t2:
            st.markdown("### Preprocessing and validation")
            st.write("The notebook compares classifiers and performs hyperparameter tuning using stratified splits. The saved fitted pipeline is loaded directly for inference; this app does not refit or alter it.")
            st.markdown("- Numeric feature count: 60")
            st.markdown("- Feature values in dataset: numeric sonar energy measurements")
            st.markdown("- Input validation in app: exact feature count, finite numbers, and values between 0 and 1")
            st.info("The app validates the expected input range. It does not perform new feature selection or PCA at prediction time.")
        with t3:
            st.markdown("### Confirmed tuned-model comparison")
            bench = pd.DataFrame([
                {"Model": "Extra Trees", "Tuning CV ROC-AUC": 0.9471},
                {"Model": "SVM", "Tuning CV ROC-AUC": 0.9434},
                {"Model": "Gradient Boosting", "Tuning CV ROC-AUC": 0.9178},
            ]).sort_values("Tuning CV ROC-AUC", ascending=False)
            st.dataframe(bench, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(bench["Model"][::-1], bench["Tuning CV ROC-AUC"][::-1])
            ax.set_xlim(0, 1); ax.set_xlabel("Best CV ROC-AUC"); ax.set_title("Hyperparameter tuning candidates"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption("These are the notebook's randomized-search CV ROC-AUC results, not hold-out ROC-AUC values.")
        with t4:
            st.markdown("### Final tuned Extra Trees evaluation")
            cols = st.columns(4)
            for col, (lab, val) in zip(cols, [
                ("Accuracy", "90.48%"), ("Mine precision", "84.62%"),
                ("Mine recall", "100.00%"), ("Mine F1", "91.67%"),
            ]):
                with col: metric_card(lab, val)
            st.markdown("The notebook's hold-out classification report gives Mine precision 0.85, recall 1.00 and F1 0.92; Rock precision 1.00, recall 0.80 and F1 0.89.")
            st.markdown("**Limitations:** the dataset is small and a random hold-out split is not an independent field trial. These scores must not be interpreted as operational detection capability.")

elif page == "Model Training":
    page_header("Model Training", "Classifier comparison, tuning and cross-validation")
    st.markdown("### Selected model")
    c1, c2, c3, c4 = st.columns(4)
    for col, (lab, val, note) in zip([c1,c2,c3,c4], [
        ("Final pipeline", "Extra Trees", "Tuned candidate saved"),
        ("Hold-out accuracy", "90.48%", "42 test samples"),
        ("Mine recall", "100.00%", "22 of 22"),
        ("ROC-AUC · tuning CV", "0.9471", "Best search score"),
    ]):
        with col: metric_card(lab, val, note)
    tabs = st.tabs(["Benchmark", "Tuning", "Cross-validation", "Model details"])
    with tabs[0]:
        st.write("The notebook explored 17 classifier candidates. The uploaded notebook's strongest tuned candidate selected for the saved final pipeline is Extra Trees.")
        bench = pd.DataFrame([
            {"Model": "Extra Trees (tuned)", "Hold-out accuracy": .9048, "Mine precision": .8462, "Mine recall": 1.0, "Mine F1": .9167},
            {"Model": "Logistic Regression (baseline)", "Hold-out accuracy": .8333, "Mine precision": .7778, "Mine recall": .9545, "Mine F1": .8571},
        ])
        st.dataframe(bench.style.format({c:"{:.4f}" for c in bench.columns if c != "Model"}), use_container_width=True)
        st.caption("Only rows with confirmed values from the notebook are shown here; this is not a fabricated full 17-model leaderboard.")
    with tabs[1]:
        tuning = pd.DataFrame([
            {"Model": "Extra Trees", "Best CV ROC-AUC": .9471, "Parameters": "500 trees; max_depth=20; max_features=None; class_weight=balanced"},
            {"Model": "SVM", "Best CV ROC-AUC": .9434, "Parameters": "RBF kernel; C=100; gamma=auto; class_weight=balanced"},
            {"Model": "Gradient Boosting", "Best CV ROC-AUC": .9178, "Parameters": "200 estimators; learning_rate=0.03; max_depth=3; subsample=0.85"},
        ])
        st.dataframe(tuning, use_container_width=True)
    with tabs[2]:
        cols = st.columns(4)
        for col, (lab, val) in zip(cols, [
            ("Mean accuracy", "83.74%"), ("Accuracy std", "6.21 pp"),
            ("Mean mine recall", "91.05%"), ("Mean ROC-AUC", "0.9334"),
        ]):
            with col: metric_card(lab, val)
        st.write("These are the tuned Extra Trees robustness cross-validation results reported by the notebook, using stratified 5-fold CV.")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(["Mean CV accuracy"], [.8374], yerr=[.0621], capsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("5-fold CV accuracy ± standard deviation"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with tabs[3]:
        if model is not None:
            st.write(f"**Loaded artifact:** `{MODEL_PATH.relative_to(BASE_DIR)}`")
            st.write(f"**Pipeline type:** `{type(model).__name__}`")
            st.write(f"**Input features:** {FEATURES}")
            st.write(f"**Classes:** {', '.join(map(str, CLASSES))}")
            st.write("**Prediction API:** `predict()` and `predict_proba()`")
        else:
            st.error(f"Model file is missing: {MODEL_PATH}")
        st.warning("This page displays notebook evaluation results; it does not train a new model on every app run.")

elif page == "Prediction System":
    page_header("Prediction System", "Give the model one sonar return and inspect its classification")
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        mode = st.radio("Input method", ["Paste 60 values", "Load a sample from dataset", "Enter manually", "Upload CSV"], horizontal=True)
        raw = ""
        if mode == "Paste 60 values":
            raw = st.text_area("60 comma-separated energy values (0–1)", placeholder="0.0200, 0.0371, 0.0428, ...", height=180, key="paste_values")
            st.caption("Enter exactly 60 values in the same feature order used for training. Commas are recommended; spaces and newlines are also accepted.")
        elif mode == "Load a sample from dataset":
            if data is None:
                st.warning("Dataset CSV not found beside app.py.")
            else:
                idx = st.number_input("Dataset row index", min_value=0, max_value=max(0, len(data)-1), value=0, step=1)
                row = data.iloc[int(idx)]
                raw = ", ".join(f"{float(v):.6f}" for v in row.drop(labels=["label"]).values)
                st.text_area("Sample values", value=raw, height=130, disabled=True)
                st.caption(f"Known dataset label for this sample: {'Mine (M)' if row['label']=='M' else 'Rock (R)'}")
        elif mode == "Enter manually":
            defaults = [0.02, 0.0371, 0.0428, 0.0207, 0.0954] + [0.1] * (FEATURES-5)
            values = []
            grid = st.columns(3)
            for i in range(FEATURES):
                with grid[i % 3]:
                    values.append(st.number_input(f"Band {i+1:02d}", min_value=0.0, max_value=1.0, value=float(defaults[i]), step=0.001, format="%.4f", key=f"manual_{i}"))
            raw = ", ".join(map(str, values))
        else:
            uploaded = st.file_uploader("Upload one-row CSV with 60 numeric columns", type=["csv"], key="predict_csv")
            if uploaded:
                try:
                    incoming = pd.read_csv(uploaded)
                    if incoming.shape[0] != 1 or incoming.shape[1] != FEATURES:
                        st.error(f"CSV must contain exactly one row and {FEATURES} feature columns.")
                    else:
                        raw = ", ".join(str(x) for x in incoming.iloc[0].tolist())
                except Exception as exc:
                    st.error(f"Could not read CSV: {exc}")
        threshold = st.slider("Mine decision threshold (probability)", min_value=0.05, max_value=0.95, value=0.55, step=0.05)
        predict_clicked = st.button("🔎 Predict", type="primary", use_container_width=True)
    # Keep the last prediction visible across Streamlit reruns, and use it to color the radar.
    if "sonar_last_result" not in st.session_state:
        st.session_state["sonar_last_result"] = None
    last_result = st.session_state["sonar_last_result"]
    with right:
        radar_markup(
            last_result["decision"] if last_result else None,
            last_result["confidence"] if last_result else None,
        )
        if last_result:
            is_mine = last_result["decision"] == "M"
            result_title = "Mine Detected" if is_mine else "Rock Detected"
            result_icon = "💣" if is_mine else "🪨"
            result_class = "mine" if is_mine else "rock"
            confidence_pct = last_result["confidence"] * 100
            st.markdown(
                f'<div class="classification-card {result_class}">'
                f'<div class="classification-kicker">Classification</div>'
                f'<div class="classification-title"><span class="classification-icon">{result_icon}</span>{result_title}</div>'
                f'<div class="classification-confidence">Confidence: <strong>{confidence_pct:.1f}%</strong> (model score)</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="probability-summary"><strong>Probability of Mine:</strong> {last_result["p_mine"]:.1%}'
                f' &nbsp;|&nbsp; <strong>Rock:</strong> {last_result["p_rock"]:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="probability-track"><div class="probability-fill" style="width:{last_result["p_mine"]*100:.2f}%;"></div></div>'
                f'<div class="probability-axis"><span>Rock</span><span>threshold {last_result["threshold"]:.2f}</span><span>Mine</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="callout prediction-helper">Choose an input, press <strong>Predict</strong>, and inspect the model output. This is an educational decision-support demo only.</div>', unsafe_allow_html=True)
    if predict_clicked:
        if model is None:
            st.error(f"Model artifact is missing. Expected path: {MODEL_PATH}")
        else:
            try:
                values = parse_values(raw)
                if len(values) != FEATURES:
                    raise ValueError(f"Expected exactly {FEATURES} values, but received {len(values)}.")
                arr = np.asarray(values, dtype=float)
                if not np.isfinite(arr).all(): raise ValueError("All values must be finite numbers.")
                if (arr < 0).any() or (arr > 1).any(): raise ValueError("Every value must be between 0 and 1.")
                frame = make_feature_frame(arr)
                probs = model.predict_proba(frame)[0]
                p_mine = float(probs[list(model.classes_).index("M")])
                p_rock = float(probs[list(model.classes_).index("R")])
                # Threshold is an explicit UI decision rule; show model's native label too.
                native = model.predict(frame)[0]
                decision = "M" if p_mine >= threshold else "R"
                # Confidence is the model probability for the selected threshold decision.
                confidence = p_mine if decision == "M" else p_rock
                st.session_state["sonar_last_result"] = {
                    "decision": decision,
                    "confidence": float(confidence),
                    "p_mine": float(p_mine),
                    "p_rock": float(p_rock),
                    "threshold": float(threshold),
                }
                st.markdown("## Mission analysis result")
                if decision == "M":
                    st.error("Threshold-based decision: MINE")
                else:
                    st.success("Threshold-based decision: ROCK")

                if decision != native:
                    st.info(
                        f"The selected threshold ({threshold:.2f}) gives "
                        f"{'MINE' if decision == 'M' else 'ROCK'}, while the "
                        f"model's default prediction is {'MINE' if native == 'M' else 'ROCK'}. "
                        "Changing the threshold trades false alarms against missed mines."
                    )

                # If the entered vector exactly matches a labelled dataset row,
                # surface its known label for transparent auditing.
                known_label = None
                matched_row = None
                if data is not None and data.shape[1] == FEATURES + 1:
                    feature_data = data.drop(columns=["label"])
                    matrix = feature_data.to_numpy(dtype=float)
                    matches = np.flatnonzero(np.all(np.isclose(matrix, arr, atol=1e-7, rtol=0), axis=1))
                    if len(matches):
                        matched_row = int(matches[0])
                        known_label = str(data.iloc[matched_row]["label"])
                        known_name = "MINE" if known_label == "M" else "ROCK"
                        predicted_name = "MINE" if decision == "M" else "ROCK"
                        if known_label == decision:
                            st.success(f"Dataset check: this matches dataset row {matched_row} (0-based), whose known label is {known_name}.")
                        else:
                            st.warning(
                                f"Dataset check: this matches dataset row {matched_row} (0-based), "
                                f"whose known label is {known_name}, but the selected-threshold result is "
                                f"{predicted_name}. This is a model misclassification, not proof that the dataset label is wrong."
                            )

                c1, c2, c3 = st.columns(3)
                with c1: metric_card("Mine probability", f"{p_mine:.1%}")
                with c2: metric_card("Rock probability", f"{p_rock:.1%}")
                with c3: metric_card("Default model label", "MINE" if native == "M" else "ROCK")
                prob_df = pd.DataFrame({"Class": ["Mine", "Rock"], "Probability": [p_mine, p_rock]}).set_index("Class")
                st.bar_chart(prob_df, y="Probability")
                sig = pd.DataFrame({"Band": np.arange(1, FEATURES+1), "Energy": arr}).set_index("Band")
                with st.expander("View input signal", expanded=False):
                    st.line_chart(sig)
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    st.caption(
        "The default mine threshold is 0.55 to reduce borderline false alarms in this demo. "
        "It is a configurable decision rule, not a newly validated operating threshold. "
        "Model probabilities are not necessarily calibrated and are not proof of an actual underwater object."
    )

else:  # About Developer
    page_header("About Developer", "Project author, tools and acknowledgements")
    a, b = st.columns([.9, 1.1], gap="large")
    with a:
        st.markdown('<div class="developer-card"><div class="feature-icon">👤</div><h2>Mayur Malaye</h2><p class="developer-role">Data Analyst · Machine Learning Project</p><p>B.E. Electronics & Telecommunication</p><p>Thane / Mumbai, India</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="portfolio-link-card"><div class="portfolio-kicker">Portfolio</div><a href="https://mayur-malaye.netlify.app/" target="_blank" rel="noopener noreferrer">Mayur\'s Portfolio ↗</a></div>', unsafe_allow_html=True)
        st.markdown("### Skills used")
        st.markdown('<div class="pill-row"><span class="pill">Python</span><span class="pill">Pandas</span><span class="pill">NumPy</span><span class="pill">scikit-learn</span><span class="pill">Matplotlib</span><span class="pill">Seaborn</span><span class="pill">Streamlit</span><span class="pill">Joblib</span></div>', unsafe_allow_html=True)
        st.markdown("### Project guide")
        st.write("Ritik Raj Vaishya")
    st.markdown("## Project credits")
    st.write("The project uses the classic UCI Sonar dataset of sonar returns from a metal cylinder (mine) and rocks. The app displays the dataset analysis and the final tuned Extra Trees evaluation recorded in the supplied notebook.")
    st.info("This is a learning and demonstration project. It is not independently validated for real-world mine detection, navigation, or clearance.")

st.markdown('<div class="footer-note"><strong>SONAR ROCK VS MINE</strong><br>Machine learning capstone · Streamlit application<br>Educational prototype — not for operational use.</div>', unsafe_allow_html=True)
