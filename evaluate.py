from ultralytics import YOLO
import numpy as np
import pandas as pd

MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)
metrics = model.val()

cm = metrics.confusion_matrix.matrix
classes = metrics.names

metrics_list = []

for i, cls in enumerate(classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (TP + TN) / (cm.sum() + 1e-6)

    metrics_list.append([
        cls, TP, TN, FP, FN,
        precision, recall, specificity, f1_score, accuracy
    ])

metrics_df = pd.DataFrame(
    metrics_list,
    columns=[
        "Class", "TP", "TN", "FP", "FN",
        "Precision", "Recall", "Specificity",
        "F1 Score", "Accuracy"
    ]
)

for column in ["Precision", "Recall", "Specificity", "F1 Score", "Accuracy"]:
    metrics_df[column + " %"] = metrics_df[column] * 100

print("\nDETAILED METRICS:\n")
print(metrics_df.to_string(index=False))

print("\n===== OVERALL MODEL PERFORMANCE =====")
print(f"Precision: {metrics_df['Precision'].mean()*100:.2f}%")
print(f"Recall: {metrics_df['Recall'].mean()*100:.2f}%")
print(f"F1 Score: {metrics_df['F1 Score'].mean()*100:.2f}%")
print(f"Accuracy: {metrics_df['Accuracy'].mean()*100:.2f}%")

metrics_df.to_csv("metrics.csv", index=False)
print("\nSaved metrics.csv")
