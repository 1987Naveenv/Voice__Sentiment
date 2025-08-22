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
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 3  # seconds
AUDIO_LENGTH = AUDIO_SAMPLE_RATE * AUDIO_DURATION

class AudioVisualDataset(Dataset):
    def __init__(self, dataframe, audio_transform=None, video_transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path, video_path, label = row.audio_path, row.video_path, row.label

        # Audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != AUDIO_SAMPLE_RATE:
            waveform = Resample(orig_freq=sr, new_freq=AUDIO_SAMPLE_RATE)(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if waveform.size(1) < AUDIO_LENGTH:
            pad_size = AUDIO_LENGTH - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
            print(f"[Padded] Audio too short: {audio_path}, Padded: {pad_size} samples")
        else:
            waveform = waveform[:, :AUDIO_LENGTH]
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        # Video
        try:
            image = Image.open(video_path).convert('RGB')
            if self.video_transform:
                image = self.video_transform(image)
        except Exception as e:
            print(f"Error loading image {video_path}: {e}")
            image = torch.zeros((3, 224, 224))

        return waveform, image, label

# ✅ Modified Comparison Model 3: FFNN + Lightweight CNN
class FFNN_CNN_AVClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Audio Path: Feedforward NN
        self.audio_branch = nn.Sequential(
            nn.Flatten(),                      # (B, 1, 48000) → (B, 48000)
            nn.Linear(48000, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # Video Path: Custom lightweight 3-layer CNN
        self.video_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 16, 112, 112)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 56, 56)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 64, 1, 1)
            nn.Flatten(),  # (B, 64)
        )

        # Fusion + Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio, video):
        audio_feat = self.audio_branch(audio)
        video_feat = self.video_branch(video)
        combined = torch.cat([audio_feat, video_feat], dim=1)
        return self.classifier(combined)

# Metadata creation
def create_metadata_from_folders(root_dir):
    classes, data = [], []
    for cls in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path): continue
        audio_folder = os.path.join(cls_path, 'audio')
        video_folder = os.path.join(cls_path, 'video')
        for file in os.listdir(audio_folder):
            if not file.endswith(".wav"): continue
            audio_path = os.path.join(audio_folder, file)
            video_name = file.replace(".wav", ".jpg")
            video_path = os.path.join(video_folder, video_name)
            if not os.path.exists(video_path): continue
            data.append((audio_path, video_path, cls))
            classes.append(cls)
    df = pd.DataFrame(data, columns=['audio_path', 'video_path', 'label'])
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    return df, le

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training Function
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=20, patience=5):
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    early_stopper = EarlyStopping(patience)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            for audio, video, labels in tqdm(dataloaders[phase]):
                audio, video, labels = audio.to(device), video.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(audio, video)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * audio.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), "best_model.pth")
                early_stopper(epoch_loss)

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    return model, history

# Evaluation
def evaluate_model(model, dataloader, device, le):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for audio, video, labels in dataloader:
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            outputs = model(audio, video)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    # Save predictions
    pd.DataFrame({"true": y_true, "pred": y_pred}).to_csv("test_predictions.csv", index=False)

    print(classification_report(y_true, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.clf()

    y_true_bin = np.eye(len(le.classes_))[y_true]
    y_prob = np.array(y_prob)

    for i, class_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} AUC: {roc_auc:.2f}")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig("roc_curves.png")
    plt.clf()

    for i, class_name in enumerate(le.classes_):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=class_name)
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig("pr_curves.png")
    plt.clf()

# Visualize training history
def plot_history(history):
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title("Training History")
    plt.savefig("training_history.png")
    plt.clf()
