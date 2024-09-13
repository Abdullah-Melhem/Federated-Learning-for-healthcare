import torch
import torch.nn.functional as F
import numpy as np


def calculate_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels).item()


def calculate_accuracy(outputs, labels):
    correct = (outputs == labels).sum().item()
    return correct / len(labels)


def calculate_f1_precision_recall(outputs, labels):
    outputs_np = outputs.cpu().numpy()
    labels_np = labels.cpu().numpy()

    tp = np.sum((outputs_np == 1) & (labels_np == 1))
    fp = np.sum((outputs_np == 1) & (labels_np == 0))
    fn = np.sum((outputs_np == 0) & (labels_np == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall


def calculate_false_alarm_rate(outputs, labels):
    outputs_np = outputs.cpu().numpy()
    labels_np = labels.cpu().numpy()

    fp = np.sum((outputs_np == 1) & (labels_np == 0))
    tn = np.sum((outputs_np == 0) & (labels_np == 0))

    return fp / (fp + tn) if (fp + tn) > 0 else 0


def calculate_roc_auc(outputs, labels):
    outputs_np = F.softmax(outputs, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    num_classes = outputs_np.shape[1]
    roc_auc = 0
    for i in range(num_classes):
        label_binary = (labels_np == i).astype(int)
        pred_proba = outputs_np[:, i]
        roc_auc += roc_auc_score_single_class(label_binary, pred_proba)
    return roc_auc / num_classes


def roc_auc_score_single_class(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return np.trapz(tpr, fpr)


def roc_curve(y_true, y_score):
    thresholds = np.sort(y_score)
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        tp = np.sum((y_score >= threshold) & (y_true == 1))
        fp = np.sum((y_score >= threshold) & (y_true == 0))
        fn = np.sum((y_score < threshold) & (y_true == 1))
        tn = np.sum((y_score < threshold) & (y_true == 0))
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr, tpr, thresholds
