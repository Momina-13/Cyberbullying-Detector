import numpy as np


def compute_metrics(y_true, y_pred):
    """
    Compute Accuracy, Precision, Recall, F1 from scratch.
    No sklearn used — all computed from TP/FP/TN/FN counts.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    accuracy  = (TP + TN) / (TP + FP + TN + FN + 1e-9)
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        'Accuracy':  round(accuracy  * 100, 2),
        'Precision': round(precision * 100, 2),
        'Recall':    round(recall    * 100, 2),
        'F1-Score':  round(f1        * 100, 2),
        'TP': int(TP), 'FP': int(FP),
        'TN': int(TN), 'FN': int(FN)
    }


def print_metrics(name, metrics):
    print(f"\n{'='*45}")
    print(f"  {name} Results")
    print(f"{'='*45}")
    print(f"  Accuracy  : {metrics['Accuracy']}%")
    print(f"  Precision : {metrics['Precision']}%")
    print(f"  Recall    : {metrics['Recall']}%")
    print(f"  F1-Score  : {metrics['F1-Score']}%")
    print(f"  TP:{metrics['TP']}  FP:{metrics['FP']}  TN:{metrics['TN']}  FN:{metrics['FN']}")
    print(f"{'='*45}")
