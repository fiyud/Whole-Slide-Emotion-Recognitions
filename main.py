import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import librosa
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import sys
from tqdm import tqdm
import random
average = 'micro'

from utils.loaddata import load_data_dual, EmoDualDataset
from train import train_model_with_auxiliary_outputs, evaluate_with_auxiliary_outputs
from utils.general import set_seed, compute_ua_wa

from arch.model import DualStreamWithAuxiliaryOutputs

set_seed(42)

data_folder = "../VNEMOS/"
waves, mfccs, labels = load_data_dual(data_folder)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMO_CLASSES = {label: i for i, label in enumerate(np.unique(labels))}

def print_classification_report(labels_preds_kfold):
    for idx, data in enumerate(labels_preds_kfold):
        print(f"K-Fold: {idx}")
        print(classification_report(data[0], data[1], target_names=EMO_CLASSES.keys()))

def average_classification_report(labels_preds_kfold):
    all_labels = []
    all_preds = []
    for data in labels_preds_kfold:
        all_labels.extend(data[0])
        all_preds.extend(data[1])
    print(classification_report(all_labels, all_preds, target_names=EMO_CLASSES.keys()))

epochs = 100
batch_size = 32
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()

acc_kfold = []
f1_kfold = []
recall_kfold = []
precision_kfold = []
data_kfold = []
labels_preds_kfold = []

for idx, (train_idx, test_idx) in enumerate(skf.split(mfccs, labels)):
    print(f"\n=== K-Fold {idx+1}/5 ===")
    
    wave_train, wave_test = waves[train_idx], waves[test_idx]
    mfcc_train, mfcc_test = mfccs[train_idx], mfccs[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    train_labels = [EMO_CLASSES[label] for label in train_labels]
    test_labels = [EMO_CLASSES[label] for label in test_labels]

    train_dataset = EmoDualDataset(wave_train, mfcc_train, train_labels)
    test_dataset = EmoDualDataset(wave_test, mfcc_test, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DualStreamWithAuxiliaryOutputs(lambda_w=0.01, lambda_aux=0.3, num_classes=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    acc = []
    f1 = []
    recall = []
    precision = []
    best_labels_preds = []

    epoch_pbar = tqdm(range(epochs), desc=f"K-Fold {idx+1}", unit="epoch")
    
    for epoch in epoch_pbar:
        loss_dict = train_model_with_auxiliary_outputs(model, train_loader, optimizer, device, max_grad_norm=1.0)
        
        avg_train_loss = loss_dict['total_loss']
        avg_ce_loss = loss_dict['main_ce_loss']
        avg_aux_loss = loss_dict['aux_loss']
        avg_wasserstein_loss = loss_dict['wasserstein_loss']

        all_labels, all_preds, aux_preds = evaluate_with_auxiliary_outputs(model, test_loader, device, use_ensemble=True)

        validation_acc = accuracy_score(all_labels, all_preds)
        recall_val = recall_score(all_labels, all_preds, average='micro')
        f1_val = f1_score(all_labels, all_preds, average='micro')
        precision_val = precision_score(all_labels, all_preds, average='micro')

        acc.append(validation_acc)
        f1.append(f1_val)
        recall.append(recall_val)
        precision.append(precision_val)

        if validation_acc > best_acc:
            best_acc = validation_acc
            best_recall = recall_val
            best_f1 = f1_val
            best_precision = precision_val
            best_labels_preds = [all_labels, all_preds]

        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "CE Loss": f"{avg_ce_loss:.4f}",
            "Aux Loss": f"{avg_aux_loss:.4f}",
            "W Loss": f"{avg_wasserstein_loss:.4f}",
            "Val Acc": f"{validation_acc:.4f}",
            "Best Acc": f"{best_acc:.4f}"
        })

        if epoch % 20 == 0:
            wave_acc = accuracy_score(all_labels, aux_preds['wave'])
            mfcc_acc = accuracy_score(all_labels, aux_preds['mfcc'])
            intermediate_acc = accuracy_score(all_labels, aux_preds['intermediate'])
            final_acc = accuracy_score(all_labels, aux_preds['final'])
            
            print(f"\nEpoch {epoch+1} Auxiliary Performance:")
            print(f"  Wave stream: {wave_acc:.4f}")
            print(f"  MFCC stream: {mfcc_acc:.4f}")
            print(f"  Intermediate: {intermediate_acc:.4f}")
            print(f"  Final: {final_acc:.4f}")
            print(f"  Ensemble: {validation_acc:.4f}")

    data_kfold.append([acc, f1, recall, precision])
    acc_kfold.append(best_acc)
    f1_kfold.append(best_f1)
    recall_kfold.append(best_recall)
    precision_kfold.append(best_precision)
    labels_preds_kfold.append(best_labels_preds)
    
    print(f"K-Fold {idx+1} Results - Best Acc: {best_acc:.4f}, Best F1: {best_f1:.4f}")

print(f"Mean Accuracy: {np.mean(acc_kfold)}")
print(f"Mean F1: {np.mean(f1_kfold)}")
print(f"Mean Precision: {np.mean(precision_kfold)}")
print(f"Mean Recall: {np.mean(recall_kfold)}")

compute_ua_wa(labels_preds_kfold)
ua, wa = compute_ua_wa(labels_preds_kfold)
print(f"UA: {ua:.4f}, WA: {wa:.4f}")