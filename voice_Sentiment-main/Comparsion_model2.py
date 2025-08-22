# -----------------------------
# Imports
# -----------------------------
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchaudio.transforms import Resample
import torchaudio
import torchvision

import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -----------------------------
# Dataset Class
# -----------------------------
class AudioVisualDataset(Dataset):
    def __init__(self, data_df, audio_dir, video_dir, transform=None):
        self.data_df = data_df
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.data_df['label'] = self.label_encoder.fit_transform(self.data_df['label'])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['audio'])
        video_path = os.path.join(self.video_dir, row['video'])

        waveform, sr = torchaudio.load(audio_path)
        resampler = Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        mfcc = torchaudio.transforms.MFCC(sample_rate=16000)(waveform)
        mfcc = mfcc.mean(dim=-1).squeeze(0)

        image = Image.open(video_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = row['label']
        return mfcc, image, label

# -----------------------------
# Model Architecture
# -----------------------------
class AVClassifierWithBiLSTMAttn(nn.Module):
    def __init__(self, num_classes):
        super(AVClassifierWithBiLSTMAttn, self).__init__()

        self.audio_lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256, 1)

        self.video_model = models.resnet18(pretrained=True)
        self.video_model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(256 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio, image):
        audio = audio.permute(0, 2, 1)
        lstm_out, _ = self.audio_lstm(audio)

        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        attn_output = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        image_features = self.video_model(image)
        combined = torch.cat([attn_output, image_features], dim=1)

        return self.classifier(combined)

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for audio, image, labels in dataloader:
            audio, image, labels = audio.to(device), image.to(device), labels.to(device)
            outputs = model(audio, image)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.clf()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.clf()

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("precision_recall_curve.png")
    plt.clf()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [] , []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        for audio, image, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            audio, image, labels = audio.to(device), image.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio, image)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for audio, image, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                audio, image, labels = audio.to(device), image.to(device), labels.to(device)
                outputs = model(audio, image)
                loss = loss_fn(outputs, labels)

                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.clf()

    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.clf()

    print("Training complete. Loss and accuracy curves saved.")
