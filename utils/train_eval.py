import torch

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from config import DEVICE
sns.set_style('whitegrid')


def fit_eval_epoch(model, criterion, dataloader, optimizer=None):
    avg_loss, corrects, total, avg_acc = 0, 0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        if optimizer is not None:
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        avg_loss += loss.item() / len(dataloader)
        _, preds = outputs.max(1)
        corrects += (preds == labels).sum().item()
        total += labels.size(0)
        avg_acc += corrects / total / len(dataloader)
    return avg_loss, avg_acc


def train(model, criterion, train_loader, test_loader,
          optimizer, scheduler, epochs):
    history = []

    for epoch in range(epochs):
        avg_loss_tr, avg_acc_tr = fit_eval_epoch(
            model, criterion, train_loader, optimizer)
        avg_loss_val, avg_acc_val = fit_eval_epoch(
            model, criterion, test_loader)
        scheduler.step()
        history.append((avg_loss_tr, avg_acc_tr, avg_loss_val, avg_acc_val))
        loss_tr, acc_tr, loss_val, acc_val = zip(*history)

        clear_output(wait=True)
        print(f'* Epoch {epoch + 1}/{epochs}')
        print(f'Loss train/val: {avg_loss_tr:.3f} | {avg_loss_val:.3f}\n'
              f'Accuracy train/val: {avg_acc_tr:.3f} | {avg_acc_val:.3f}')

        fig = plt.figure(figsize=(17.2, 5))

        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.plot(loss_tr, label='train_loss')
        plt.plot(loss_val, label='val_loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.plot(acc_tr, label='train_acc')
        plt.plot(acc_val, label='val_acc')
        plt.legend()
        plt.show()

    return history


def test(model):
    model.eval()
    with torch.no_grad():
        corrects, total = 0, 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            corrects += (preds == labels).sum().item()
            total += labels.size(0)

    return corrects / total
