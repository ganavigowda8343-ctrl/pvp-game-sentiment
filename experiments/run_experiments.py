"""
Experiment Pipeline and Benchmark Runner
Replicates the experimental evaluation from:
"Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications"
Stanford CS229 (Spring 2021)

Authors: Ganavi Gowda (PES1UG24CS707), Aadarsh Reddy (PES1UG24CS706)
Institution: PES University
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import clean_text, PvPFeatureExtractor
from src.models import get_binary_svm_models, get_multiclass_ml_models
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_model_comparison

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/pvp_sentiment_dataset.csv"))
os.makedirs(ASSETS_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("  TROLLS, HATERS, OR TEAMMATES: CLASSIFYING SENTIMENT IN PVP GAME COMMUNICATIONS")
    print("  Student Researchers: Ganavi Gowda (PES1UG24CS707), Aadarsh Reddy (PES1UG24CS706)")
    print("  Institution: PES University | Reference: Stanford CS229")
    print("=" * 80)

    # 1. Load Data
    print(f"\n[1] Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total messages: {len(df):,}")

    # Plot Class Distribution (Matching Figure 2 from CS229 paper)
    plt.figure(figsize=(6, 4))
    palette = ["#4a90e2", "#50e3c2", "#e25050"]
    ax = sns.countplot(x="sentiment", data=df, palette=palette, hue="sentiment", legend=False)
    plt.title("Figure 2: Balance of Classes from the Dataset", fontsize=11, fontweight="bold")
    plt.xlabel("Sentiment Class", fontsize=10)
    plt.ylabel("Message Count", fontsize=10)
    plt.tight_layout()
    dist_path = os.path.join(ASSETS_DIR, "class_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to: {dist_path}")

    # 2. Text Preprocessing
    print("\n[2] Preprocessing chat text with gamer slang handling...")
    df["clean_text"] = df["message"].apply(lambda t: clean_text(t, expand_slang=False))

    # 3. Train-Test Split (80/20 matching CS229 Section 3)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    print(f"Training set: {len(X_train_raw)} samples | Test set: {len(X_test_raw)} samples")

    # 4. Feature Extraction (TF-IDF ~2700 dimensions & 165-term dictionary baseline)
    print("\n[3] Extracting features (TF-IDF & 165-term dictionary baseline)...")
    extractor = PvPFeatureExtractor(max_features=2700, ngram_range=(1, 2))
    extractor.fit_tfidf(X_train_raw)

    X_train_tfidf = extractor.transform_tfidf(X_train_raw)
    X_test_tfidf = extractor.transform_tfidf(X_test_raw)

    X_train_dict = extractor.transform_dict_baseline(X_train_raw)
    X_test_dict = extractor.transform_dict_baseline(X_test_raw)

    print(f"TF-IDF Feature Space Dimension: {X_train_tfidf.shape[1]}")
    print(f"Dictionary Baseline Feature Dimension: {X_train_dict.shape[1]}")

    # 5. Baseline: Binary SVM (Section 4.1 & Table 2)
    print("\n" + "=" * 60)
    print("  SECTION 5.1: BASELINES (BINARY SVM & NAIVE BAYES)")
    print("=" * 60)
    # Binary mapping: Class 0 -> 0 (Positive/Neutral), Classes 1 & 2 -> 1 (Negative/Derogatory)
    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    svm_results = []
    for name, model in get_binary_svm_models().items():
        model.fit(X_train_tfidf, y_train_bin)
        preds = model.predict(X_test_tfidf)
        acc = calculate_metrics(y_test_bin, preds)["accuracy"]
        svm_results.append({"Model": name, "Accuracy": acc})
        print(f"  {name:30s} Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Dictionary-based Naive Bayes baseline
    nb_dict = MultinomialNB()
    nb_dict.fit(X_train_dict, y_train)
    preds_nb_dict = nb_dict.predict(X_test_dict)
    acc_nb_dict = calculate_metrics(y_test, preds_nb_dict)["accuracy"]
    svm_results.append({"Model": "Naive Bayes (165-term Dict Baseline)", "Accuracy": acc_nb_dict})
    print(f"  {'Naive Bayes (165-term Baseline)':30s} Accuracy: {acc_nb_dict:.4f} ({acc_nb_dict*100:.2f}%)")

    # Save Confusion matrix for Naive Bayes baseline (Figure 5)
    cm_nb_path = os.path.join(ASSETS_DIR, "confusion_matrix_naive_bayes_baseline.png")
    plot_confusion_matrix(
        y_test, preds_nb_dict,
        "Figure 5: Confusion Matrix for Naive Bayes Baseline (165 Terms)",
        cm_nb_path
    )

    # 6. Multi-Class ML Classifiers (Section 5.2, Table 3 & Table 4)
    print("\n" + "=" * 60)
    print("  SECTION 5.2: MULTI-CLASS ML CLASSIFIERS (TF-IDF)")
    print("=" * 60)
    ml_models = get_multiclass_ml_models()
    ml_metrics_list = []

    best_model_name = None
    best_acc = 0.0
    best_preds = None

    for name, model in ml_models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        metrics = calculate_metrics(y_test, preds)
        
        ml_metrics_list.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "F1 Class 0 (Neutral/Pos)": metrics["f1_class_0"],
            "F1 Class 1 (Negative)": metrics["f1_class_1"],
            "F1 Class 2 (Derogatory)": metrics["f1_class_2"],
            "Macro F1": metrics["macro_f1"]
        })
        print(f"  {name:38s} | Acc: {metrics['accuracy']:.4f} | F1 (0/1/2): {metrics['f1_class_0']:.2f} / {metrics['f1_class_1']:.2f} / {metrics['f1_class_2']:.2f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_model_name = name
            best_preds = preds

    ml_df = pd.DataFrame(ml_metrics_list)

    # Plot Best Model Confusion Matrix
    cm_best_path = os.path.join(ASSETS_DIR, "confusion_matrix_best_model.png")
    plot_confusion_matrix(
        y_test, best_preds,
        f"Confusion Matrix: Best Performing Model ({best_model_name})",
        cm_best_path
    )

    # Plot Model Accuracies Comparison Bar Chart
    acc_chart_path = os.path.join(ASSETS_DIR, "model_accuracies_comparison.png")
    plot_model_comparison(
        ml_df, "Accuracy",
        "Comparison of Machine Learning Classifiers (TF-IDF Encodings)",
        acc_chart_path
    )

    # Plot F1 Score Breakdown
    f1_melted = ml_df.melt(
        id_vars=["Model"],
        value_vars=["F1 Class 0 (Neutral/Pos)", "F1 Class 1 (Negative)", "F1 Class 2 (Derogatory)"],
        var_name="Class", value_name="F1 Score"
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(x="Model", y="F1 Score", hue="Class", data=f1_melted, palette="Set2")
    plt.title("Per-Class F1 Score Breakdown Across Models", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Model", fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.70, 1.05)
    plt.legend(title="Sentiment Class", loc="lower right")
    plt.tight_layout()
    f1_chart_path = os.path.join(ASSETS_DIR, "f1_score_breakdown.png")
    plt.savefig(f1_chart_path, dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print("  ALL EXPERIMENTS COMPLETE! PLOTS AND ASSETS SAVED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    main()
