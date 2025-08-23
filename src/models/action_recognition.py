import os
import random
import warnings

import timm
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as t

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Config:
    IMAGE_SIZE = 224
    CLIP_LENGTH = 16
    TIMM_MODEL_NAME = 'efficientnet_b0'
    LSTM_HIDDEN_SIZE = 512
    LSTM_NUM_LAYERS = 2
    DROPOUT = 0.4

cfg = Config()

CLASS_NAMES = [
    'backhand',
    'backhand2hands',
    'backhand_slice',
    'backhand_volley',
    'flat_service',
    'forehand_flat',
    'forehand_openstands',
    'forehand_slice',
    'forehand_volley',
    'kick_service',
    'slice_service',
    'smash'
]


class VideoActionModel(nn.Module):
    def __init__(self, num_classes, model_name=cfg.TIMM_MODEL_NAME, pretrained=False):
        super().__init__()
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        feature_dim = self.feature_extractor.num_features

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=cfg.LSTM_HIDDEN_SIZE,
            num_layers=cfg.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.LSTM_HIDDEN_SIZE, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        features = self.feature_extractor(x)
        features = features.mean(dim=[-1, -2])

        features = features.view(b, t, -1)

        lstm_out, (h_n, c_n) = self.lstm(features)

        last_hidden_state = h_n[-1]

        return self.classifier(last_hidden_state)

class ActionRecognizer:
    def __init__(self, model_path, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        self.model = VideoActionModel(num_classes=len(CLASS_NAMES))

        state_dict = torch.load(model_path, map_location=self.device)
        cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned_state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transforms = self._get_transforms()

    def _get_transforms(self):
        temp_model = timm.create_model(cfg.TIMM_MODEL_NAME, pretrained=True)
        data_config = timm.data.resolve_data_config(model=temp_model)
        mean, std = data_config['mean'], data_config['std']

        return t.Compose([
            t.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            t.ConvertImageDtype(torch.float32),
            t.Normalize(mean=mean, std=std),
        ])

    def _preprocess_video(self, video_path):
        vid = torchvision.io.VideoReader(video_path, "video")
        metadata = vid.get_metadata()
        total_frames = int(metadata['video']['duration'][0] * metadata['video']['fps'][0])

        max_start = max(0, total_frames - cfg.CLIP_LENGTH)
        start_frame = random.randint(0, max_start) if max_start > 0 else 0
        start_time = start_frame / metadata['video']['fps'][0]

        vid.seek(start_time)
        video_frames = []
        for i, frame in enumerate(vid):
            if i >= cfg.CLIP_LENGTH:
                break
            video_frames.append(self.transforms(frame['data']))

        while len(video_frames) < cfg.CLIP_LENGTH:
            video_frames.append(video_frames[-1])

        video_tensor = torch.stack(video_frames, 0).unsqueeze(0)
        return video_tensor

    def predict(self, video_path):
        input_tensor = self._preprocess_video(video_path)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score

if __name__ == '__main__':
    MODEL_SAVE_PATH = "model.pt"

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at '{MODEL_SAVE_PATH}'")
    else:
        recognizer = ActionRecognizer(model_path=MODEL_SAVE_PATH)

        base_video_dir = "thetis_data/VIDEO_RGB"
        try:
            random_class_name = random.choice(CLASS_NAMES)
            class_folder_path = os.path.join(base_video_dir, random_class_name)

            video_files = [f for f in os.listdir(class_folder_path) if f.endswith('.avi')]

            if not video_files:
                 print(f"Error: No .avi video files found in '{class_folder_path}'")
                 sample_video_path = None
            else:
                random_video_file = random.choice(video_files)
                sample_video_path = os.path.join(class_folder_path, random_video_file)

        except (FileNotFoundError, IndexError):
            print(f"Error: Could not find or select a random video from '{base_video_dir}'. Please ensure the data exists.")
            sample_video_path = None

        if sample_video_path:
            print(f"\nPerforming inference on randomly selected video: {sample_video_path}")

            predicted_action, confidence = recognizer.predict(sample_video_path)

            if predicted_action:
                print(f"\n--> Predicted Action: {predicted_action}")
                print(f"--> Confidence: {confidence:.2%}")
