import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import sys

CONF_TOL = 1e-6


def load_preds(path):
    df = pd.read_csv(path)
    return df.set_index("file_name").sort_index()


def load_gt(path):
    df = pd.read_csv(path, header=None, names=["file_name", "true_label"])
    return df.set_index("file_name").sort_index()


def compare(csv1, csv2, gt_csv):

    df1 = load_preds(csv1)
    df2 = load_preds(csv2)
    gt = load_gt(gt_csv)

    # Align everything
    common = df1.index.intersection(df2.index).intersection(gt.index)
    df1 = df1.loc[common]
    df2 = df2.loc[common]
    gt = gt.loc[common]

    # Find prediction differences
    label_diff = df1["label"] != df2["label"]
    conf_diff = ~np.isclose(df1["confidence"], df2["confidence"], atol=CONF_TOL)

    diff_mask = label_diff

    diff_df = pd.DataFrame({
        "file_name": common,
        "pred1": df1["label"],
        "pred2": df2["label"],
        "confidence1": df1["confidence"],
        "confidence2": df2["confidence"],
        "true_label": gt["true_label"]
    })[diff_mask]

    print(f"\n⚠️ Found {len(diff_df)} differing rows\n")
    print(diff_df.head(30))

    # Confusion matrices
    y_true = gt["true_label"]

    y1 = df1["label"]
    y2 = df2["label"]

    labels = sorted(y_true.unique())

    cm1 = confusion_matrix(y_true, y1, labels=labels)
    cm2 = confusion_matrix(y_true, y2, labels=labels)

    acc1 = accuracy_score(y_true, y1)
    acc2 = accuracy_score(y_true, y2)

    f1_score_1 = f1_score(y_true, y1, pos_label="Nodule")
    f1_score_2 = f1_score(y_true, y2, pos_label="Nodule")

    print("\n==============================")
    print("CSV1 Confusion Matrix")
    print("==============================")
    print(cm1)
    print(f"Accuracy: {acc1:.4f}")
    print(f"F1 Score: {f1_score_1:.4f}")
    print("\n==============================")
    print("CSV2 Confusion Matrix")
    print("==============================")
    print(cm2)
    print(f"Accuracy: {acc2:.4f}")
    print(f"F1 Score: {f1_score_2:.4f}")

    if acc1 > acc2:
        print("\n✅ CSV1 is closer to ground truth")
    elif acc2 > acc1:
        print("\n✅ CSV2 is closer to ground truth")
    else:
        print("\n⚠️ Both models have identical accuracy")


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage:")
        print("python compare_with_gt.py preds1.csv preds2.csv ground_truth.csv")
        sys.exit(1)

    compare(sys.argv[1], sys.argv[2], sys.argv[3])