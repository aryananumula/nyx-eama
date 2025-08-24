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

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="yte_6Bh7ZBVN" outputId="09dfd102-712d-41af-e01c-41dc3b0ce65a"
# !pip install torchcodec av certifi charset-normalizer contourpy==1.3.2 cycler==0.12.1 einops==0.8.1 ezc3d==1.5.19 filelock==3.18.0 fonttools==4.59.0 fsspec==2025.7.0 hf-xet==1.1.5 huggingface-hub==0.33.4 idna==3.10 iniconfig==2.1.0 jinja2==3.1.6 joblib==1.5.1 kiwisolver==1.4.8 markupsafe==3.0.2 matplotlib==3.10.3 mpmath==1.3.0 networkx==3.5 numpy==2.3.1 packaging==25.0 pandas==2.3.1 pillow==11.3.0 pluggy==1.6.0 pygments==2.19.2 pyparsing==3.2.3 pytest==8.4.1 python-dateutil==2.9.0.post0 pytz==2025.2 pyyaml==6.0.2 regex==2024.11.6 requests==2.32.4 safetensors==0.5.3 scikit-learn==1.7.1 scipy==1.16.0 seaborn==0.13.2 setuptools==80.9.0 six==1.17.0 sympy==1.14.0 threadpoolctl==3.6.0 tokenizers==0.21.2 torch==2.7.1 torchvision==0.22.1 tqdm==4.67.1 transformers==4.53.3 typing-extensions==4.14.1 tzdata==2025.2 urllib3==2.5.0 vit-pytorch==1.10.1

# %% colab={"base_uri": "https://localhost:8080/"} id="BlYJUdkdkRZW" outputId="775a1133-f898-4549-fe83-32676d877fef"
# !pip install vit_pytorch pytorchvideo

# %% id="bQro6Cf6xJBl"
import logging
import os
import random
import shutil

import matplotlib.pyplot as plt
import requests
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as t
from torchvision.datasets.folder import make_dataset
from tqdm.auto import tqdm
from vit_pytorch.vivit import ViT
from typing import List, Tuple, Optional
import torchvision

# %% id="EoRknrCpxWdh"
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class to centralize parameters
class Config:
    # Data parameters
    IMAGE_SIZE = 160
    CLIP_LENGTH = 64
    BATCH_SIZE = 512
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 2

    # Model parameters
    IMAGE_PATCH_SIZE = 16
    FRAME_PATCH_SIZE = 8
    MODEL_DIM = 512
    SPATIAL_DEPTH = 6
    TEMPORAL_DEPTH = 6
    MODEL_HEADS = 8
    MLP_DIM = 1024
    DROPOUT = 0.4


cfg = Config()

# %% colab={"base_uri": "https://localhost:8080/", "height": 53} id="XhXCi8Jo_dmM" outputId="cba61f55-ed00-41b3-84d2-a912745c7a92"
# from google.colab import drive
# drive.mount('/content/drive')
# shutil.move('/content/drive/MyDrive/thetis_data', "/content/thetis_data")

# %% colab={"base_uri": "https://localhost:8080/", "height": 67, "referenced_widgets": ["4fcbf4aff533404b94c7304bbc567345", "ea65982843d14c23be0499441fc14408", "0732a53e01af4d5a8c1e969bc68c7154", "f3f4fd8f6ce24077a25134b71382eea0", "b5dac7686d724679b5f22dd6e0f3a005", "ba781248defb4a81927852973e6c71e0", "f6cb50cc93394a69b8db479d3c18e693", "b11fbca459f041b6b674d73e034e645f", "c61b7b93fce7436a93370979d78428c8", "ea58ff00a6684664a0794a2382d7030e", "49974d952f2d4cd486106fe2b61c981e", "3a9c80d5ceec47c08511fa57fd96b003", "2418d9d2e33c452983fbcf5faca84009", "11151e0cce444a0c913ccbb170c0dc6e", "ec91518c6e52431cac678187dda71a4e", "96299b1445cd4d25afe3e0e1adc1ced3", "f6a3c9e2892044c996678dd633d5c1d3", "b0b060d0cac34709bb6b0d691c4e5c6c", "f3a4bee927b946b9b269ac2f5739c58b", "a1a6c09026c44931be716c5ea0fcf4f6", "67bacdd91901489aae62cab53451e3d8", "f7a019c052594798b34225b96e41a13f", "f94df02798e54b1b8445a842c49baf37", "93e35db84c19447fbfbfbab2dd931ddc", "425c20941f864eab9d1ebca7961203d9", "b3a3d1002b4945b4b3b45e44eff04fef", "0d02592b23464397aa29d422d92dc883", "46f0fceecb804ec4abd57130fc6c1053", "581e07c2904e48feb3b6709df0ef7fe7", "a959ebc85474440597ba2d7379e1e493", "23572858b61a45d98d5452c95de3084f", "7e7b7b51a0d5449ea18f3e6c8f0ace40", "62c6400230184799909b27203ab3b359", "cb6e721d3f674a0eb2a5edb4c4dfbd8e", "88e3d4e8162149909cc1ba8b537cb81c", "fd7e998d28194b2c8d4692fd57f6d0c4", "086258ce23fe4d7f8b8208f1e7533592", "3e14aac2e82a4c51bda5ccdc0398a7e8", "e86c2fe56f6f4c64b198900cabd13d97", "0698f1e1c95c405f9ffe993590bdbf89", "6d093ec03d4f4c3fb8a60075bb309a5f", "a93643577eee45b3949887d042883278", "825b228746464d1ab2891347786f171c", "bad151c686ff4261a44bb5b1b76876e5", "d6f080053dde4607bf3d4ef3862076b5", "5bc8fe5bcd9849ac9777f372286dcd56", "8faccc23ee5246399e200c4333bbace0", "724a2b9afe6849c7ab7c69604d88533e", "392cd20f2a4e49c684932ad9e8d46150", "8a040a91f6e643859a5f9cb9a4aaa76d", "2b770c66916b4f72918eaf46a21d29c4", "84fccff791cd43189938995000938b0c", "dde1c56b73ad4f8289f78a578d039fdc", "7903400cd7f0441286edd7d0b92ff6a1", "dfd2a4f18c1349cd835d1e25a5c5c975", "22919eab15084c5289d5e6de35575911", "7376daf8ee794fe38c082fb566dffcbb", "d313e92aba9841ae86c576b3d8c8d228", "9912f36cb6c540089becbaca676d3967", "f0391f5f2a054b68b5074a3f8451a448", "17e4f1fb754c469bb81ff1ed8ad75701", "13982637561045a7ba39fee8089b042f", "ccd4c7c9ff024006b74bf459b1250aed", "0b8e76a77ad342d58a3cd915454c89a7", "9b1a633296b349fc92ca941b72c5d5f3", "588617f5c8d144daa1edd27764827deb", "c0c550abbcc84bb9baeae6693ff963b5", "b2b14532d90b42348687649f3fc8de74", "14cc6dee6544441e8336c3b5442f04f3", "990aedbb062d4746860e0e666d29b82b", "7c2e7092463f49df9f771c556592c450", "6e10903471674aa1ae0382df67ab9ddd", "ffc675bf49d348f294c8ba59efb556da", "704fc06f50a3401586e8a7de17fca6ab", "244b6dae582442a7a30344b9c724fe46", "1c9a841dcf5e4e89bc8670c4be71b893", "5b6aa4e6ed4541e3bcbb8323bcd9ef8f", "344f646b1e9442d8b5544f8ad328411f", "81382401411a41fa833e414f365160b9", "a76d97de4d2040fa94434e48766ed753", "d497cef44e6c4951a03a40239d96dabb", "a5c54466f260417ea3ba790618c2d3fd", "5e2f7ef45d974f8390a6d22416b4f44b", "4b6ea72153164c26a00dab39429fb493", "a3161dd0f7444daa8aff362d312e4442", "bbe16ea86caf4c4b874fc35e72b920ee", "37f167cfeab64e149e55f5bccd61c7bb", "6afdd721e0a349069287f74c87f74c08", "9867af12e783458fb351531a18146b8f", "36001e8b76f2472cbb94cc4a5993c8b9", "2c64a15a0deb43c3a200c8dd49446b0d", "742343584fb34767b57b6307951fae22", "3ef9eb271ae748aba871081de5cd7022", "792955fe67f144e4907fed9ab2de2cd9", "93926871e6554b5c9a1fd896a57f92f5", "515a414e407441af98ae876487854cca", "26ce170a91de418dbe9b873ede38fb23", "51135002c2ca48f58d0a2856a7396982", "9ed4d9e952c0442f9663f74f8d34775c", "86453c812f5f45a3881c13b00e012868", "9c4b61c88ec343b2bfcdfa180e2636d7", "5948150a1f154f89a8325d003e6b6ce7", "91e9d624f1514ac8b02b905311b59827", "124a266d1c3d4bf6960a9f3f185e2f63", "75cff580072a4268aece95ae2aa76830", "1c94177918c742af89119b7326d317b5", "f67692ec05444719a1d59e7edc8410ad", "432730999a4945e791063a82c1a940be", "b1d010b7c4b24674ac2ea6c39f447e3f", "555add5d72f54cd394e268c8ef75b2a1", "cebb8fd9ed0a4f9c9fc5b7a17c9dfb70", "5b1b62a486f74b74aa47e6897ead13d5", "c27bcbb2a90945d9a2583ec23d3ac156", "efe408ae60b44e38b84f6d89a81038b3", "882eb7eca0f64ecc9f21ef0cd4ac9d39", "894864894a6f40639901c57aac379f88", "0bb0fc9a0313437db1dcffc73f03b058", "145500af29984f4d9070ff3e416663ab", "26a759fa549f40928809e530f99bf628", "9df15aa15e0248cdafe76518c09c0193", "e1d9840db91a45aaacea7409cd011f2b", "2a9e02c687cf4b849e780708271b3e1a", "804fc5702bc74ae780fa95bee9debec3", "cda8db61fd1a481fb093ec86fcedb494", "64d962e371f6476fad5068840e1ded80", "3e70e81cfeee4e4694a8c33a63547694", "24be3c67b79240f998fd8d0ffc2c94f7", "a583b09327304a3d8bd7e6c362a37b40", "8330a5a53fdf4fb19792d28194478942", "acb6ec5b3e534474918e464b3625f0b9", "e4e1ccd5d0174ddfa31354c3d773c631", "16790cebb6634f60b087b2bbb802ac8c", "3738b5bcbe014bc48ce32b87ab962626", "d2553c8f4c464694b6b1953176390dba", "9c4b152b442b4703837498f397982192", "21f0720251204465b98f2a4eab8b43f8", "ad5e091884d24d5e8146425091152ac0", "a375a862bb114784886a404708ed7808", "bfb19b764cab442e9d28a864f2f7a2b0", "85237a4e427e4e0db9bc29ac19fbc9d9", "a1e36867728845abb9db4a1fe114aadf", "23d6a39023094acebcaf1699cb1e0832", "f9227fe2623443719593dc3a9df5f134"]} id="yVNHv6ZudH-L" outputId="82649fd0-cd69-4c63-88e8-9ea0f5f0917f"
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


# %% colab={"base_uri": "https://localhost:8080/"} id="pp2gOHPS4jqt" outputId="c94b44c8-7ba4-4c84-96da-8e7c3fb192f1"
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

            try:
                vid = torchvision.io.VideoReader(path, "video")
                metadata = vid.get_metadata()
                video_frames = []

                # FIXED: Random temporal sampling instead of always starting from beginning
                # Get video duration and randomly sample start time
                duration = metadata['video']['duration'][0]
                fps = metadata['video']['fps'][0]
                total_frames = int(duration * fps)

                # Calculate maximum start frame to ensure we have enough frames
                max_start_frame = max(0, total_frames - self.clip_len)
                start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0
                start_time = start_frame / fps

                # Seek to the random start time
                vid.seek(start_time)
                current_pts = start_time

                # Collect frames
                frame_count = 0
                for frame in vid:
                    if frame_count >= self.clip_len:
                        break
                    video_frames.append(self.frame_transform(frame['data']))
                    current_pts = frame['pts']
                    frame_count += 1

                # If we didn't get enough frames, pad with the last frame
                while len(video_frames) < self.clip_len:
                    video_frames.append(video_frames[-1] if video_frames else torch.zeros(3, 160, 160))

                video = torch.stack(video_frames, 0)
                if self.video_transform:
                    video = self.video_transform(video)

                output = {
                    'path': path,
                    'video': video,
                    'target': target,
                    'start': start_time,
                    'end': current_pts
                }
                yield output

            except Exception as e:
                print(f"Error loading video {path}: {e}")
                # Skip this sample and try another one
                continue


transforms = [
    t.Resize((160, 160)),
    t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
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

print("Checking class distribution...")
class_counts = {}
for path, target in dataset.samples:
    class_name = _find_classes(f"thetis_data/{base_folder}")[0][target]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1
print("Class distribution:", class_counts)

# %%
# Error tracking lists
train_errors = []
val_errors = []
epochs_completed = []

# For confusion matrix
all_val_targets = []
all_val_preds = []

num_epochs = 50
best_val_acc = 0.0

# %%
model = ViT(
    image_size=cfg.IMAGE_SIZE,
    frames=cfg.CLIP_LENGTH,
    image_patch_size=cfg.IMAGE_PATCH_SIZE,
    frame_patch_size=cfg.FRAME_PATCH_SIZE,
    num_classes=len(_find_classes(f"thetis_data/{base_folder}")[0]),
    dim=cfg.MODEL_DIM,
    spatial_depth=cfg.SPATIAL_DEPTH,
    temporal_depth=cfg.TEMPORAL_DEPTH,
    heads=cfg.MODEL_HEADS,
    mlp_dim=cfg.MLP_DIM,
    variant='factorized_encoder',
    dropout=cfg.DROPOUT,
)

try:
    model.load_state_dict(torch.load("models/latest_model.pt"))
except:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
model = model.to(device)
model = torch.compile(model)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="GoQP5h6V9kBF" outputId="4baeea80-96d3-4194-9c60-e2c8f82b8186"
batch_size = cfg.BATCH_SIZE

train_samples, val_samples = train_test_split(dataset.samples, test_size=0.2, random_state=42, stratify=[sample[1] for sample in dataset.samples])

train_dataset = Dataset(f"thetis_data/{base_folder}", epoch_size=len(train_samples), frame_transform=frame_transform, clip_len=frame_size)
train_dataset.samples = train_samples

val_dataset = Dataset(f"thetis_data/{base_folder}", epoch_size=len(val_samples), frame_transform=val_frame_transform, clip_len=frame_size)
val_dataset.samples = val_samples

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=cfg.NUM_WORKERS,
    prefetch_factor=cfg.PREFETCH_FACTOR,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=cfg.NUM_WORKERS,
)

LEARNING_RATE = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

# Loss function
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

scaler = GradScaler('cuda')

# Memory usage monitoring function
def print_memory_usage():
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {gpu_allocated:.2f}GB, Reserved: {gpu_reserved:.2f}GB")


# %%
try:
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples_count = 0
        train_correct = 0
        batch_count = 0
        print("here")

        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            videos = batch['video'].to(device, non_blocking=True)
            videos = videos.permute(0, 2, 1, 3, 4)  # (B, frames, channels, H, W) -> (B, channels, frames, H, W)
            targets = batch['target'].long().to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, targets)

            print("loss calc")

            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

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
                with autocast('cuda'):
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

        # Update best validation accuracy and save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "models/best_model.pt");


        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1}/{num_epochs}] - LR: {current_lr:.6f}, Train Loss: {train_loss/train_samples_count:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, '
              f'Train Error: {train_error_rate:.4f}, Val Error: {val_error_rate:.4f}')
        with open("logs.txt", "a") as f:
            print(f'Epoch [{epoch+1}/{num_epochs}] - LR: {current_lr:.6f}, Train Loss: {train_loss/train_samples_count:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, '
              f'Train Error: {train_error_rate:.4f}, Val Error: {val_error_rate:.4f}', file=f)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), f'models/latest_model.pt')

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

# %%
