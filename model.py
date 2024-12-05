import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


data = []
with open("splice.data", "r") as file:
    for line in file:
        label, _, sequence = line.strip().split(",")
        label = 1 if (label == "EI" or label == "IE") else 0
        data.append((sequence.strip(), label))


sequences, labels = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42, stratify=labels
)


def fdtf_encoding(sequence, decoy_train, max_length=None):
    paired_counts_seq = {}
    paired_counts_decoy = {}

    # Count paired-nucleotide frequencies
    for i in range(len(sequence) - 1):
        pair = sequence[i:i+2]
        paired_counts_seq[pair] = paired_counts_seq.get(pair, 0) + 1

    for decoy_seq in decoy_train:
        for i in range(len(decoy_seq) - 1):
            pair = decoy_seq[i:i+2]
            paired_counts_decoy[pair] = paired_counts_decoy.get(pair, 0) + 1

    # Calculate differences
    all_pairs = set(paired_counts_seq.keys()).union(paired_counts_decoy.keys())
    differences = [paired_counts_seq.get(pair, 0) - paired_counts_decoy.get(pair, 0) for pair in all_pairs]

    # Fix length by padding or truncating
    if max_length is None:
        max_length = len(differences)

    if len(differences) < max_length:
        differences.extend([0] * (max_length - len(differences)))
    else:
        differences = differences[:max_length]

    return differences

# Compute max_length using the training set
decoy_train = [seq for seq, label in zip(X_train, y_train) if label == 0]
max_length = max(len(fdtf_encoding(seq, decoy_train)) for seq in X_train)

# Encode data with fixed length
X_train_encoded = np.array([fdtf_encoding(seq, decoy_train, max_length) for seq in X_train])
X_test_encoded = np.array([fdtf_encoding(seq, decoy_train, max_length) for seq in X_test])

# Initialize models
models = {
    "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)
    y_prob = model.predict_proba(X_test_encoded)[:, 1] if hasattr(model, "predict_proba") else y_pred

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "Confusion Matrix": cm
    })

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")


plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_encoded)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test_encoded)
    else:
        y_prob = model.predict(X_test_encoded)  # fallback for models like KNN if no proba

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()