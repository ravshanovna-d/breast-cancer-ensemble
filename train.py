"""Compare single models with voting, stacking and blending ensembles on the
Wisconsin breast cancer data, then export the stacking pipeline for the web demo.

Usage:  python train.py
Outputs: web/model.onnx, web/meta.json, web/assets/*.png
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from export_utils import check_onnx, examples, export_onnx, feature_schema, write_meta

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
SEED = 42
N_FEATURES = 10


def load_data():
    """Wisconsin Diagnostic Breast Cancer data (ships with scikit-learn).
    Target: 1 = malignant, 0 = benign."""
    d = load_breast_cancer(as_frame=True)
    X = d.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    y = (d.target == 0).astype(int)  # scikit-learn encodes malignant as 0
    return X, y


def base_estimators():
    return [
        ("lr", LogisticRegression(max_iter=5000, C=1.0)),
        ("svm", SVC(C=3.0, kernel="rbf", probability=True, random_state=SEED)),
        ("knn", KNeighborsClassifier(n_neighbors=9)),
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1)),
        ("gb", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=SEED)),
    ]


class BlendingClassifier(ClassifierMixin, BaseEstimator):
    """Blending: base models are trained on one part of the training data, and a
    logistic regression learns how to combine their probabilities on a held-out part."""

    def __init__(self, estimators, holdout=0.25, random_state=None):
        self.estimators = estimators
        self.holdout = holdout
        self.random_state = random_state

    def fit(self, X, y):
        X_fit, X_hold, y_fit, y_hold = train_test_split(
            X, y, test_size=self.holdout, stratify=y, random_state=self.random_state)
        self.fitted_ = [clone(e).fit(X_fit, y_fit) for _, e in self.estimators]
        self.meta_ = LogisticRegression().fit(self._level1(X_hold), y_hold)
        self.classes_ = self.meta_.classes_
        return self

    def _level1(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted_])

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._level1(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def build_models():
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    names = {"lr": "Logistic Regression", "svm": "SVM (RBF)", "knn": "k-NN",
             "rf": "Random Forest", "gb": "Gradient Boosting"}
    models = {names[k]: scaled(m) for k, m in base_estimators()}
    models["Soft Voting"] = scaled(VotingClassifier(base_estimators(), voting="soft"))
    models["Blending"] = scaled(BlendingClassifier(base_estimators(), random_state=SEED))
    models["Stacking"] = Pipeline([("scale", StandardScaler()), ("model", StackingClassifier(
        base_estimators(), final_estimator=LogisticRegression(C=1.0),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), stack_method="predict_proba"))])
    return models


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    X_all, y = load_data()
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, stratify=y, random_state=SEED)

    # 1) Pick 10 complementary measurements with forward selection (training data only)
    sfs = SequentialFeatureSelector(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), n_features_to_select=N_FEATURES,
        direction="forward", scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    sfs.fit(X_train_all, y_train)
    features = list(X_all.columns[sfs.get_support()])
    print("Selected features:", features)
    X_train, X_test = X_train_all[features], X_test_all[features]

    # 2) Compare single models and ensembles: 5-fold CV repeated 3 times (small dataset)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    models = build_models()
    rows = []
    for name, model in models.items():
        r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                           scoring={"ROC-AUC": "roc_auc", "F1": "f1", "Recall": "recall", "Accuracy": "accuracy"})
        rows.append({"model": name, **{k: float(np.mean(r[f"test_{k}"])) for k in ("ROC-AUC", "F1", "Recall", "Accuracy")}})
        print(f"{name:20s} ROC-AUC={rows[-1]['ROC-AUC']:.4f}  F1={rows[-1]['F1']:.4f}  Recall={rows[-1]['Recall']:.4f}")
    comparison = sorted(rows, key=lambda r: -r["ROC-AUC"])

    best_name = "Stacking"
    best = models[best_name].fit(X_train, y_train)

    # 3) Final evaluation on the untouched test set (all approaches, for the figure)
    test_auc = {}
    for name, model in models.items():
        m = model if name == best_name else clone(model).fit(X_train, y_train)
        test_auc[name] = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Recall (malignant)": recall_score(y_test, pred),
        "Precision (malignant)": precision_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
    print("Test:", {k: round(v, 4) for k, v in test_metrics.items()})
    print("Test ROC-AUC per approach:", {k: round(v, 4) for k, v in test_auc.items()})

    # 4) Figures
    sns.set_theme(style="whitegrid")
    comp = pd.DataFrame(comparison).set_index("model")["ROC-AUC"].sort_values()
    ens = {"Soft Voting", "Blending", "Stacking"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    comp.plot.barh(ax=ax, color=["#4f46e5" if m in ens else "#a5b4fc" for m in comp.index])
    ax.set_xlim(comp.min() - 0.01, 1.0); ax.set_xlabel("Mean CV ROC-AUC (5-fold × 3)"); ax.set_ylabel("")
    ax.set_title("Single models (light) vs ensembles (dark)")
    fig.tight_layout(); fig.savefig(ASSETS / "model_comparison.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Benign", "Malignant"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, name="Stacking", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "roc_curve.png", dpi=130); plt.close(fig)

    stack = best.named_steps["model"]
    names = {"lr": "Logistic Regression", "svm": "SVM (RBF)", "knn": "k-NN", "rf": "Random Forest", "gb": "Gradient Boosting"}
    w = pd.Series(stack.final_estimator_.coef_[0], index=[names[n] for n, _ in stack.estimators]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    w.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Meta-model coefficient")
    ax.set_title("How much the stacker trusts each base model")
    fig.tight_layout(); fig.savefig(ASSETS / "stacking_weights.png", dpi=130); plt.close(fig)

    # 5) Export for the browser and verify it matches scikit-learn
    export_onnx(best, X_train, WEB / "model.onnx")
    err = check_onnx(WEB / "model.onnx", best, X_test)
    print(f"ONNX check OK (max diff {err:.2e})")

    stat = {"mean": "average over all cell nuclei in the image",
            "error": "standard error across nuclei",
            "worst": "average of the 3 largest nuclei"}

    def label(c):
        if c.startswith("mean_"):
            return "Mean " + c[5:].replace("_", " ")
        if c.startswith("worst_"):
            return "Worst " + c[6:].replace("_", " ")
        return c[:-6].replace("_", " ").capitalize() + " (std. error)"

    labels = {c: label(c) for c in features}
    helps = {c: stat["mean" if c.startswith("mean_") else "worst" if c.startswith("worst_") else "error"] for c in features}
    write_meta(WEB / "meta.json", {
        "title": "Breast Cancer Ensemble Classifier",
        "subtitle": ("Classifies a breast tumour as benign or malignant from 10 measurements of cell nuclei "
                     "in a biopsy image, using a stacked ensemble of five models."),
        "tags": ["Classification", "Ensembles", "Stacking", "Blending", "scikit-learn", "ONNX"],
        "task": "classification",
        "classes": [0, 1], "positive_class": 1,
        "output_label": "Predicted probability the tumour is malignant",
        "verdicts": ["Looks benign", "Uncertain — needs expert review", "Looks malignant"],
        "thresholds": [0.3, 0.7],
        "result_note": ("Educational demo only — <b>not medical advice</b> and not a diagnostic tool. "
                        "Try “Random example” to load real cases from the test set."),
        "features": feature_schema(X_train, labels, helps),
        "examples": examples(X_test),
        "model_name": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metric_formats": {k: "percent" for k in test_metrics if k != "ROC-AUC"},
        "metrics_note": f"Held-out test set: {len(X_test)} tumours (positive class = malignant).",
        "figures": [
            {"src": "assets/model_comparison.png", "caption": "Cross-validated ROC-AUC: single models vs voting, blending and stacking."},
            {"src": "assets/stacking_weights.png", "caption": "The stacking meta-model learns how much to trust each base model."},
            {"src": "assets/roc_curve.png", "caption": "ROC curve of the stacking model on the test set."},
            {"src": "assets/confusion_matrix.png", "caption": "Correct and wrong predictions on the test set."},
        ],
        "steps": [
            f"Used the Wisconsin Diagnostic Breast Cancer data ({len(X_all)} tumours, 30 measurements, malignant = positive class).",
            f"Picked {N_FEATURES} complementary measurements with forward feature selection on the training set only.",
            "Trained 5 diverse base models: Logistic Regression, SVM, k-NN, Random Forest, Gradient Boosting.",
            "Combined them three ways: soft voting, blending (hold-out meta-model) and stacking (out-of-fold meta-model).",
            "Compared everything with 5-fold cross-validation repeated 3 times, then checked once on a held-out 20% test set.",
            "Exported the full stacking pipeline to ONNX so it runs in the browser.",
        ],
        "dataset": {"name": "Breast Cancer Wisconsin (Diagnostic)",
                    "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic", "rows": len(X_all)},
        "repo_url": "https://github.com/ravshanovna-d/breast-cancer-ensemble",
        "notebook_url": "https://github.com/ravshanovna-d/breast-cancer-ensemble/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    })
    print("Wrote web/model.onnx and web/meta.json")


if __name__ == "__main__":
    main()
