from __future__ import annotations


def character_accuracy(pred_texts: list[str], true_texts: list[str]) -> float:
    total_chars = 0
    correct_chars = 0

    for pred, truth in zip(pred_texts, true_texts):
        min_len = min(len(pred), len(truth))
        correct_chars += sum(1 for i in range(min_len) if pred[i] == truth[i])
        total_chars += len(truth)

    if total_chars == 0:
        return 0.0
    return correct_chars / total_chars


def full_plate_accuracy(pred_texts: list[str], true_texts: list[str]) -> float:
    if not true_texts:
        return 0.0
    correct = sum(1 for p, t in zip(pred_texts, true_texts) if p == t)
    return correct / len(true_texts)


def advanced_metrics(pred_texts: list[str], true_texts: list[str]) -> dict[str, float]:
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_true = []
        y_pred = []
        for p, t in zip(pred_texts, true_texts):
            max_len = max(len(p), len(t))
            p_padded = p.ljust(max_len, '_')
            t_padded = t.ljust(max_len, '_')
            y_true.extend(list(t_padded))
            y_pred.extend(list(p_padded))
            
        return {
            "precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        }
    except ImportError:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
