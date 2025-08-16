import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
import torchvision.io
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as t
from torchvision.datasets.folder import make_dataset
from tqdm.auto import tqdm
from vit_pytorch.vit_3d import ViT

repo_owner = "THETIS-dataset"
repo_name = "dataset"
base_folder = "VIDEO_RGB"
local_base = f"thetis_data/{base_folder}"

folders = []

# Make sure base local target exists
os.makedirs(local_base, exist_ok=True)
try:
    # Get action subdirectories
    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{base_folder}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(github_api_url, headers=headers)
    folders = [item for item in response.json() if item["type"] == "dir"]

    # For each action subfolder (like 'backhand'), download .avi files
    for folder in tqdm(folders, desc="Subfolders"):
        # build subfolder API path
        folder_name = folder["name"]
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{base_folder}/{folder_name}"
        r = requests.get(api_url, headers=headers)
        files = [f for f in r.json() if f["name"].endswith(".avi")]
        local_folder = os.path.join(local_base, folder_name)
        os.makedirs(local_folder, exist_ok=True)
        for file in tqdm(files, desc=f"Downloading {folder_name}", leave=False):
            raw_url = file["download_url"]
            dest_path = os.path.join(local_folder, file["name"])
            if not os.path.exists(dest_path):  # skip if already downloaded
                file_content = requests.get(raw_url).content
                with open(dest_path, "wb") as f:
                    f.write(file_content)

    print(f"All {base_folder} .avi files downloaded to thetis_data/")
except Exception as e:
    print(f"Error fetching data from GitHub: {e}")

def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(Dataset).__init__()
        self.samples = get_samples(root)

        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size
        self.clip_len = clip_len
        self.frame_transform = frame_transform if frame_transform is not None else lambda x: x
        self.video_transform = video_transform

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for i in range(self.epoch_size):
            path, target = random.choice(self.samples)
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []

            # Start from the beginning of the video (start=0)
            start = 0.0
            current_pts = start

            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']

            while len(video_frames) < self.clip_len:
                video_frames.append(video_frames[-1] if video_frames else torch.zeros(3, 160, 160))

            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)

            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts
            }
            yield output

transforms = [
    t.Resize((160, 160)),
    t.RandomHorizontalFlip(p=0.3),
    t.ColorJitter(brightness=0.1, contrast=0.1),
    t.ConvertImageDtype(torch.float32),
    t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]
frame_transform = t.Compose(transforms)

val_transforms = [
    t.Resize((160, 160)),
    t.ConvertImageDtype(torch.float32),
    t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]
val_frame_transform = t.Compose(val_transforms)

frame_size = 64

dataset = Dataset(f"thetis_data/{base_folder}", epoch_size=None, frame_transform=frame_transform, clip_len=frame_size)

batch_size = 16

model = ViT(
    image_size=160,
    image_patch_size=16,
    frames=frame_size,
    frame_patch_size=4,
    num_classes=len(_find_classes(f"thetis_data/{base_folder}")[0]),
    dim=256,
    depth=4,
    heads=8,
    mlp_dim=512,
    dropout=0.2,
    emb_dropout=0.2
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
model = model.to(device)

train_samples, val_samples = train_test_split(dataset.samples, test_size=0.2, random_state=42)

train_dataset = Dataset(f"thetis_data/{base_folder}", epoch_size=len(train_samples), frame_transform=frame_transform, clip_len=frame_size)
train_dataset.samples = train_samples

val_dataset = Dataset(f"thetis_data/{base_folder}", epoch_size=len(val_samples), frame_transform=val_frame_transform, clip_len=frame_size)
val_dataset.samples = val_samples

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
)

base_lr = 1e-4
scaled_lr = base_lr * (batch_size / 8)

optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=scaled_lr * 3,
    epochs=50,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos'
)

# Loss function
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization

# ADDED: Mixed precision training
scaler = GradScaler()

# Error tracking lists
train_errors = []
val_errors = []
epochs_completed = []

# For confusion matrix
all_val_targets = []
all_val_preds = []

num_epochs = 50
best_val_acc = 0.0

# Memory usage monitoring function
def print_memory_usage():
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {gpu_allocated:.2f}GB, Reserved: {gpu_reserved:.2f}GB")

try:
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples_count = 0
        train_correct = 0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            videos = batch['video'].to(device, non_blocking=True)
            videos = videos.permute(0, 2, 1, 3, 4)  # (B, frames, channels, H, W) -> (B, channels, frames, H, W)
            targets = batch['target'].long().to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.item() * targets.size(0)
            train_samples_count += targets.size(0)

            # Calculate training accuracy
            predicted = outputs.argmax(dim=1)
            train_correct += (predicted == targets).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}] completed with {batch_count} batches')

        # Calculate training error rate
        train_accuracy = train_correct / train_samples_count
        train_error_rate = 1 - train_accuracy
        train_errors.append(train_error_rate)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        epoch_val_targets = []
        epoch_val_preds = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                videos = batch['video'].to(device, non_blocking=True)
                videos = videos.permute(0, 2, 1, 3, 4)
                targets = batch['target'].long().to(device, non_blocking=True)

                # Use mixed precision for validation too
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                predicted = outputs.argmax(dim=1)

                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

                # Store predictions and targets for confusion matrix
                epoch_val_targets.extend(targets.cpu().numpy())
                epoch_val_preds.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / total_predictions
        val_accuracy = correct_predictions / total_predictions
        val_error_rate = 1 - val_accuracy
        val_errors.append(val_error_rate)
        epochs_completed.append(epoch + 1)

        # Store final epoch predictions for confusion matrix
        if epoch == num_epochs - 1:
            all_val_targets = epoch_val_targets
            all_val_preds = epoch_val_preds

        # Update best validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1}/{num_epochs}] - LR: {current_lr:.6f}, Train Loss: {train_loss/train_samples_count:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, '
              f'Train Error: {train_error_rate:.4f}, Val Error: {val_error_rate:.4f}')

        # Plot error rates every 5 epochs instead of every epoch to reduce clutter
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_completed, train_errors, 'b-', label='Training Error Rate', linewidth=2)
            plt.plot(epochs_completed, val_errors, 'r-', label='Validation Error Rate', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Error Rate')
            plt.title('Training and Validation Error Rates Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(1, num_epochs)
            plt.ylim(0, max(max(train_errors), max(val_errors)) * 1.1)
            plt.show()

except KeyboardInterrupt:
    print("\nTraining interrupted by user")

finally:
    # Clean up memory
    if 'videos' in locals():
        del videos
    if 'targets' in locals():
        del targets
    if 'outputs' in locals():
        del outputs
    if 'loss' in locals():
        del loss

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("Memory cleaned up")

# Generate final confusion matrix
if all_val_targets and all_val_preds:
    class_names = _find_classes(f"thetis_data/{base_folder}")[0]
    cm = confusion_matrix(all_val_targets, all_val_preds)

    plt.figure(figsize=(12, 10))
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cmd.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title('Confusion Matrix - Final Validation Results', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print classification metrics
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_accuracy = cm[i, i] / cm[i].sum()
            print(f"{class_name}: {class_accuracy:.3f}")

print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.4f}')

# Final error rates plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_completed, train_errors, 'b-', label='Training Error Rate', linewidth=2)
plt.plot(epochs_completed, val_errors, 'r-', label='Validation Error Rate', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.title('Final Training and Validation Error Rates')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(epochs_completed, [1-err for err in train_errors], 'b-', label='Training Accuracy', linewidth=2)
plt.plot(epochs_completed, [1-err for err in val_errors], 'r-', label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
