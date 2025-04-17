import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_classification_model(y_true, y_pred, class_names=['A', 'B', 'C', 'D', 'F']):
    
    # 1. Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")

    # 2. Precision, Recall, F1-Score (per class)
    print("\nPer-Class Metrics:")
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)

    for i, class_name in enumerate(class_names):
        print(f"{class_name} - Precision: {metrics[0][i]:.3f}, Recall: {metrics[1][i]:.3f}, F1: {metrics[2][i]:.3f}")

    # 3. Macro Averages
    macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(f"\nMacro Avg - Precision: {macro[0]:.3f}, Recall: {macro[1]:.3f}, F1: {macro[2]:.3f}")

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()