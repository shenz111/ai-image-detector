import torch


def accuracy(pred, target):
    """准确率"""
    return (pred.argmax(1) == target).float().mean().item()


def precision_recall_f1(pred, target):
    """精确率、召回率、F1 (二分类)"""
    pred_cls = pred.argmax(1)
    tp = ((pred_cls == 1) & (target == 1)).sum().item()
    fp = ((pred_cls == 1) & (target == 0)).sum().item()
    fn = ((pred_cls == 0) & (target == 1)).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


def compute_metrics(pred, target):
    acc = accuracy(pred, target)
    prec, rec, f1 = precision_recall_f1(pred, target)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}