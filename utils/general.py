import random
import numpy as np
import torch
import os
from sklearn.metrics import classification_report, accuracy_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_ua_wa(labels_preds_kfold):
    true_labels = np.array([])
    predicted_labels = np.array([])

    for data in labels_preds_kfold:
        true_labels = np.append(true_labels, data[0])
        predicted_labels = np.append(predicted_labels, data[1])

    ua = accuracy_score(true_labels, predicted_labels)

    wa = []
    for label in np.unique(true_labels):
        i_true = true_labels[true_labels == label]
        i_predicted = predicted_labels[true_labels == label]
        wa.append(np.sum(i_true == i_predicted) / len(i_true))
    wa = np.mean(wa)
    return ua, wa