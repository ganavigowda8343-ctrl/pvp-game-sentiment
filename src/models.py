"""
Model Implementations and Classifiers Module
Project: Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications
Authors: Ganavi Gowda (PES1UG24CS707), Aadarsh Reddy (PES1UG24CS706)
Institution: PES University
Reference: Stanford CS229 Machine Learning Project (Spring 2021)
"""

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_binary_svm_models():
    """
    Binary SVM baselines as described in CS229 Section 4.1:
    Tests Linear, RBF, and Polynomial kernels separating Neutral (0) vs Toxic (1 & 2).
    """
    return {
        "SVM (Linear Kernel)": SVC(kernel="linear", C=1.0, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "SVM (Polynomial Kernel)": SVC(kernel="poly", degree=3, C=1.0, random_state=42)
    }

def get_multiclass_ml_models():
    """
    Multi-class ML Classifiers as described in CS229 Section 4.2 - 4.4:
    - Multinomial Naive Bayes (TF-IDF)
    - K-Nearest Neighbors (K = 1, 2, 3, 6)
    - Random Forests (Gini vs Entropy, Max Depth variations)
    - Multi-layer Perceptron (Feedforward Neural Network)
    """
    return {
        "Naive Bayes (TF-IDF)": MultinomialNB(alpha=1.0),
        "KNN (K=1)": KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
        "KNN (K=2)": KNeighborsClassifier(n_neighbors=2, metric="euclidean"),
        "KNN (K=3)": KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
        "KNN (K=6)": KNeighborsClassifier(n_neighbors=6, metric="euclidean"),
        "Random Forest (Gini, Unconstrained)": RandomForestClassifier(
            n_estimators=100, criterion="gini", max_depth=None, random_state=42
        ),
        "Random Forest (Entropy, Unconstrained)": RandomForestClassifier(
            n_estimators=100, criterion="entropy", max_depth=None, random_state=42
        ),
        "Random Forest (Entropy, max_depth=32)": RandomForestClassifier(
            n_estimators=100, criterion="entropy", max_depth=32, random_state=42
        ),
        "Feedforward NN (MLP ReLU)": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=250, random_state=42
        )
    }
