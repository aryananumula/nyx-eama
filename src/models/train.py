# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# !pip install torch torchvision torchaudio
# !pip install timm torchcodec av certifi charset-normalizer contourpy cycler einops ezc3d filelock fonttools fsspec hf-xet huggingface-hub idna iniconfig jinja2 joblib kiwisolver markupsafe matplotlib mpmath networkx numpy packaging pandas pillow pluggy pygments pyparsing pytest python-dateutil pytz pyyaml regex requests safetensors scikit-learn scipy seaborn setuptools six sympy threadpoolctl tokenizers tqdm transformers typing-extensions tzdata urllib3

# %%
import logging
import os
import random

# %%
import warnings

import matplotlib.pyplot as plt
import requests
import timm
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms as t
from torchvision.datasets.folder import make_dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration optimized for fine-tuning a pretrained model on a small dataset
class Config:
    # Data parameters
    IMAGE_SIZE = 224 # Use the standard size for the chosen timm model (EfficientNet-B0)
    CLIP_LENGTH = 16 # Shorter clips can be effective and train faster
    BATCH_SIZE = 16  # Smaller batch size for a larger model to fit in memory
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 2

    # Model parameters
    TIMM_MODEL_NAME = 'efficientnet_b0'
    LSTM_HIDDEN_SIZE = 512
    LSTM_NUM_LAYERS = 2
    DROPOUT = 0.4

cfg = Config()

# %%
# Data downloading section (same as before)
repo_owner = "THETIS-dataset"
repo_name = "dataset"
base_folder = "VIDEO_RGB"
local_base = f"thetis_data/{base_folder}"

os.makedirs(local_base, exist_ok=True)
try:
    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{base_folder}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(github_api_url, headers=headers)
    response.raise_for_status()
    folders = [item for item in response.json() if item["type"] == "dir"]

    for folder in tqdm(folders, desc="Downloading Class Folders"):
        folder_name = folder["name"]
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{base_folder}/{folder_name}"
        r = requests.get(api_url, headers=headers)
        r.raise_for_status()
        files = [f for f in r.json() if f["name"].endswith(".avi")]
        local_folder = os.path.join(local_base, folder_name)
        os.makedirs(local_folder, exist_ok=True)
        for file in tqdm(files, desc=f"Downloading {folder_name}", leave=False):
            raw_url = file["download_url"]
            dest_path = os.path.join(local_folder, file["name"])
            if not os.path.exists(dest_path):
                file_content = requests.get(raw_url).content
                with open(dest_path, "wb") as f:
                    f.write(file_content)
    print(f"All {base_folder} .avi files downloaded to thetis_data/")
except Exception as e:
    print(f"Error fetching data from GitHub: {e}")

# %%
def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class VideoIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, clip_len=16):
        super().__init__()
        self.samples = get_samples(root)
        self.epoch_size = len(self.samples) if epoch_size is None else epoch_size
        self.clip_len = clip_len
        self.frame_transform = frame_transform if frame_transform is not None else lambda x: x

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            path, target = random.choice(self.samples)
            try:
                vid = torchvision.io.VideoReader(path, "video")
                metadata = vid.get_metadata()
                video_frames = []
                total_frames = int(metadata['video']['duration'][0] * metadata['video']['fps'][0])

                max_start_frame = max(0, total_frames - self.clip_len)
                start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0
                start_time = start_frame / metadata['video']['fps'][0]

                vid.seek(start_time)
                for i, frame in enumerate(vid):
                    if i >= self.clip_len: break
                    video_frames.append(self.frame_transform(frame['data']))

                while len(video_frames) < self.clip_len:
                    if video_frames: video_frames.append(video_frames[-1])
                    else: video_frames.append(torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

                yield {'video': torch.stack(video_frames, 0), 'target': target}
            except Exception as e:
                logger.warning(f"Skipping video due to error: {path}, {e}")
                continue

# Get normalization stats for the timm model
temp_model = timm.create_model(cfg.TIMM_MODEL_NAME, pretrained=True)
data_config = timm.data.resolve_data_config(model=temp_model)
mean = data_config['mean']
std = data_config['std']

train_transforms = t.Compose([
    t.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    t.ConvertImageDtype(torch.float32),
    t.Normalize(mean=mean, std=std),
])

val_transforms = t.Compose([
    t.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    t.ConvertImageDtype(torch.float32),
    t.Normalize(mean=mean, std=std),
])

dataset_root = f"thetis_data/{base_folder}"
all_samples = get_samples(dataset_root)
class_names, _ = _find_classes(dataset_root)
num_classes = len(class_names)

# %%
# --- MODEL DEFINITION ---
# This class wraps a pretrained timm model and adds temporal modeling layers.
class VideoActionModel(nn.Module):
    def __init__(self, num_classes, model_name=cfg.TIMM_MODEL_NAME, pretrained=True):
        super().__init__()
        # 1. Load a pretrained image model from timm as the feature extractor
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the original classifier
            global_pool=''  # Return features before pooling
        )
        # Get the output feature dimension of the backbone
        feature_dim = self.feature_extractor.num_features

        # 2. Add a temporal modeling layer (LSTM) to process the sequence of features
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=cfg.LSTM_HIDDEN_SIZE,
            num_layers=cfg.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0
        )

        # 3. Add a final classifier to predict the action
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.LSTM_HIDDEN_SIZE, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W) -> B=batch, T=frames, C=channels, H=height, W=width
        b, t, c, h, w = x.shape

        # Reshape to process all frames at once
        x = x.view(b * t, c, h, w)

        # Extract features from each frame
        features = self.feature_extractor(x) # Shape: (B*T, feature_dim, H', W')
        features = features.mean(dim=[-1, -2]) # Global Average Pooling -> (B*T, feature_dim)

        # Reshape back to a sequence for the LSTM
        features = features.view(b, t, -1) # Shape: (B, T, feature_dim)

        # Process sequence with LSTM
        # We only need the output of the last time step
        lstm_out, (h_n, c_n) = self.lstm(features)

        # Use the last hidden state of the top layer for classification
        last_hidden_state = h_n[-1] # Shape: (B, LSTM_HIDDEN_SIZE)

        # Get final predictions
        return self.classifier(last_hidden_state)


# Initialize model and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
model = VideoActionModel(num_classes=num_classes).to(device)

try:
    state_dict = torch.load('models/best_model.pt', weights_only=True)
    cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
except Exception:
    print('starting from scratch')

model = torch.compile(model)
print(f"Using device: {device}")
print(f"Loaded timm model: {cfg.TIMM_MODEL_NAME}")

# %%
# --- TRAINING SETUP ---
train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42, stratify=[s[1] for s in all_samples])

train_dataset = VideoIterableDataset(dataset_root, epoch_size=len(train_samples), frame_transform=train_transforms, clip_len=cfg.CLIP_LENGTH)
train_dataset.samples = train_samples

val_dataset = VideoIterableDataset(dataset_root, epoch_size=len(val_samples), frame_transform=val_transforms, clip_len=cfg.CLIP_LENGTH)
val_dataset.samples = val_samples

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=True, prefetch_factor=cfg.PREFETCH_FACTOR)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=True)

# Optimizer, Scheduler, and Loss
LEARNING_RATE = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler('cuda')

# %%
# --- TRAINING LOOP ---
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

num_epochs = 50
best_val_acc = 0.0
train_errors, val_errors, epochs_completed = [], [], []

try:
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_samples_count = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            videos = batch['video'].to(device, non_blocking=True) # B, T, C, H, W
            targets = batch['target'].long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * targets.size(0)
            predicted = outputs.argmax(dim=1)
            train_correct += (predicted == targets).sum().item()
            train_samples_count += targets.size(0)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{train_correct / train_samples_count:.4f}'})

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_samples_count = 0.0, 0, 0
        epoch_val_targets, epoch_val_preds = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                videos = batch['video'].to(device, non_blocking=True)
                targets = batch['target'].long().to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                predicted = outputs.argmax(dim=1)
                val_correct += (predicted == targets).sum().item()
                val_samples_count += targets.size(0)
                epoch_val_targets.extend(targets.cpu().numpy())
                epoch_val_preds.extend(predicted.cpu().numpy())

        # Logging, Saving, and Plotting
        avg_train_loss = train_loss / train_samples_count
        avg_val_loss = val_loss / val_samples_count
        val_accuracy = val_correct / val_samples_count

        train_errors.append(1 - (train_correct / train_samples_count))
        val_errors.append(1 - val_accuracy)
        epochs_completed.append(epoch + 1)

        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")

        log_msg = f'Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
        print(log_msg)
        with open("logs/training_log.txt", "a") as f: f.write(log_msg + '\n')

        if (epoch + 1) % 10 == 0: torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), 'models/latest_model.pt')

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:
    print(f'\nTraining finished! Best validation accuracy: {best_val_acc:.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_completed, train_errors, 'b-o', label='Training Error Rate')
    plt.plot(epochs_completed, val_errors, 'r-o', label='Validation Error Rate')
    plt.xlabel('Epoch'); plt.ylabel('Error Rate'); plt.title('Training vs. Validation Error Rate')
    plt.legend(); plt.grid(True); plt.show()

    if 'epoch_val_targets' in locals() and epoch_val_targets:
        cm = confusion_matrix(epoch_val_targets, epoch_val_preds)
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 10))
        cmd.plot(cmap='Blues', xticks_rotation='vertical', ax=ax)
        plt.title('Final Validation Confusion Matrix'); plt.show()

# %%
