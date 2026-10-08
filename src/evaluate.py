"""
Evaluation and Metrics Visualization Module
Project: Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications
Authors: Ganavi Gowda (PES1UG24CS707), Aadarsh Reddy (PES1UG24CS706)
Institution: PES University
Reference: Stanford CS229 Machine Learning Project (Spring 2021)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

CLASS_NAMES = ["Neutral/Positive (0)", "Negative Attitude (1)", "Derogatory (2)"]

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_class_0": f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        "f1_class_1": f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        "f1_class_2": f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".3f", cmap="Blues",
        xticklabels=["Class 0", "Class 1", "Class 2"],
        yticklabels=["Class 0", "Class 1", "Class 2"]
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(results_df, metric_col, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    sns.barplot(
        x="Model", y=metric_col, data=results_df,
        palette="viridis", hue="Model", legend=False
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classifier Model", fontsize=11)
    plt.ylabel(metric_col, fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.70, 1.02)
    
    for i, v in enumerate(results_df[metric_col]):
        plt.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
