import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn
from utils import threshold_tensor

def evaluate(data_loader, model, config):
    device = config['device']
    real, pred = [], []
    total_loss = 0
    with torch.no_grad():
        model.eval()
        for student_ids, question_ids, labels in data_loader:
            student_ids = student_ids.to(device)
            question_ids = question_ids.to(device)
            labels = threshold_tensor(labels).to(device).float()
            real += labels.tolist()

            output = model(student_ids, question_ids).view(-1)
            criterion = nn.BCELoss()
            batch_loss = criterion(output, labels)
            total_loss += batch_loss
            pred += output.tolist()
        model.train()

    real, pred = np.array(real), np.array(pred)
    auc = roc_auc_score(real, pred)
    return {'auc': auc, 'loss': total_loss / len(data_loader)}

def train(train_loader, val_loader, model, config):
    device = config['device']
    lr = config['learning_rate']
    epochs = config['num_epochs']
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc, best_step, wait = 0, 0, 100
    results = {'train_loss': [], 'val_auc': [], 'train_auc': [], 'val_loss': []}

    for ep in range(1, epochs + 1):
        train_loss = 0
        for student_ids, question_ids, labels in train_loader:
            student_ids = student_ids.to(device)
            question_ids = question_ids.to(device)
            labels = threshold_tensor(labels).to(device).float()

            optimizer.zero_grad()
            pred = model(student_ids, question_ids).view(-1)
            criterion = nn.BCELoss()
            batch_loss = criterion(pred, labels)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.data.item()

        train_metrics = evaluate(train_loader, model, config)
        val_metrics = evaluate(val_loader, model, config)

        results['train_loss'].append(train_loss / len(train_loader))
        results['train_auc'].append(train_metrics['auc'])
        results['val_auc'].append(val_metrics['auc'])
        results['val_loss'].append(val_metrics['loss'].item())

        if val_metrics['auc'] > best_auc:
            best_auc, best_step = val_metrics['auc'], ep

        if best_step + wait <= ep:
            break

    return results
