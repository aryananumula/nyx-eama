import itertools
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as t
from torchvision.datasets.folder import make_dataset
from vit_pytorch.vit_3d import ViT


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(RandomDataset).__init__()

        self.samples = get_samples(root)

        # Allow for temporal jittering
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
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            current_pts = start
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts}
            yield output

transforms = [
    t.Resize((192, 192)),
    t.ConvertImageDtype(torch.float32),  # Convert from uint8 to float32
]
frame_transform = t.Compose(transforms)

dataset = RandomDataset("thetis_data/VIDEO_Skelet3D", epoch_size=None, frame_transform=frame_transform)

loader = DataLoader(dataset, batch_size=4)
data = {"video": [], 'start': [], 'end': [], 'tensorsize': []}

df = pd.DataFrame(data)

model = ViT(
    image_size = 192,          # image size
    image_patch_size = 32,     # image patch size (increased to reduce patches)
    frames = 16,               # number of frames
    frame_patch_size = 2,      # frame patch size (changed back to 2)
    num_classes = len(_find_classes("thetis_data/VIDEO_Skelet3D")[0]), # number of classes for classification
    dim = 512,                 # reduced dimension
    depth = 4,                 # reduced depth
    heads = 8,                 # number of heads
    mlp_dim = 1024,           # reduced dimension of feedforward network
    dropout = 0.1,
    emb_dropout = 0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_samples, val_samples = train_test_split(dataset.samples, test_size=0.2, random_state=42)

# Create separate datasets
train_dataset = RandomDataset("thetis_data/VIDEO_Skelet3D", epoch_size=len(train_samples), frame_transform=frame_transform)
train_dataset.samples = train_samples

val_dataset = RandomDataset("thetis_data/VIDEO_Skelet3D", epoch_size=len(val_samples), frame_transform=frame_transform)
val_dataset.samples = val_samples

train_loader = DataLoader(train_dataset, batch_size=4)
val_loader = DataLoader(val_dataset, batch_size=4)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        videos = batch['video'].to(device)  # Shape: (batch_size, frames, channels, height, width)
        # Rearrange to (batch_size, channels, frames, height, width) for ViT 3D
        videos = videos.permute(0, 2, 1, 3, 4)
        targets = batch['target'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')

    scheduler.step()

    # Print epoch statistics
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')

    # Validation
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    for batch_idx, batch in enumerate(val_loader):
        videos = batch['video'].to(device)  # Shape: (batch_size, frames, channels, height, width)
        # Rearrange to (batch_size, channels, frames, height, width) for ViT 3D
        videos = videos.permute(0, 2, 1, 3, 4)
        targets = batch['target'].to(device)

        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, targets)

        predicted = outputs.argmax(dim=1)

        total_predictions += targets.size(0)

        correct_predictions += (predicted == targets).sum()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Validation Loss: {loss.item():.4f}')

    validation_accuracy = correct_predictions / total_predictions
    avg_loss = running_loss / total_predictions
    print(f'Validation for epoch [{epoch+1}/{num_epochs}] completed. Average Validation Loss: {avg_loss:.4f}. Validation Accuracy: {validation_accuracy}')
