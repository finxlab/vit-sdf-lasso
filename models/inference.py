import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_32
import torchvision.transforms as transforms

# ================== Transform ==================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

# ================== Dataset ==================
class CustomDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        with Image.open(img_path) as image:
            img = np.array(image)
        if self.transform:
            img = self.transform(img)
        ticker = img_path.split('/')[-1].split('.')[0]
        date = img_path.split('/')[-2]
        return img, ticker, date

# ================== CNN (Xiu et al.) ==================
class CNN20(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(3, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(3, 1)),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(46080, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 1, 64, 60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 46080)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

# ================== ViT (Byun et al.)==================
def create_vit_model(num_encoder_layers=2, num_classes=2):
    model = vit_b_32(weights=None)
    model.encoder.layers = nn.Sequential(*[model.encoder.layers[i] for i in range(num_encoder_layers)])
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

# ================== Load Models (5 seeds) ==================
def load_models(model_type, weights_dir, device):
    """Load 5 seed models"""
    seeds = ['seed1', 'seed42', 'seed123', 'seed999', 'seed12345']
    models = []
    
    for seed in seeds:
        weight_path = os.path.join(weights_dir, seed, 'best_model.pth')
        
        if model_type == 'vit':
            model = create_vit_model(num_encoder_layers=2)
            state_dict = torch.load(weight_path, map_location=device)
            
            # Fix state dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                if "encoder.layers.encoder_layer_" in k:
                    k = k.replace("encoder.layers.encoder_layer_", "encoder.layers.")
                if "heads.head.0" in k:
                    k = k.replace("heads.head.0", "heads.head")
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        
        elif model_type == 'cnn':
            model = CNN20()
            model.load_state_dict(torch.load(weight_path, map_location=device))
        
        model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded {seed}")
    
    return models

# ================== Main ==================
if __name__ == '__main__':

    MODEL_TYPE = 'vit'  # 'vit' or 'cnn'

    WEIGHTS_DIR = 'models/weights/' + MODEL_TYPE
    if MODEL_TYPE == 'vit':
        IMAGE_DIR = 'models/weights/images/rgb/*'
    elif MODEL_TYPE == 'cnn':
        IMAGE_DIR = 'models/weights/images/gray/*'
    OUTPUT_DIR='pred/'  + MODEL_TYPE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading {MODEL_TYPE.upper()} models...")
    models = load_models(MODEL_TYPE, WEIGHTS_DIR, device)
    print(f"Loaded {len(models)} models!")
    
    image_paths = glob(IMAGE_DIR)
    image_paths.sort()
    print(f"Found {len(image_paths)} images")
    
    dataset = CustomDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print("Generating predictions...")
    all_probs = []
    tickers, dates = [], []
    
    for i, model in enumerate(models):
        print(f"  Model {i+1}/5...")
        probs = []
        
        with torch.no_grad():
            for images, ticker, date in tqdm(dataloader):
                images = images.to(device)
                outputs = model(images)
                probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy().tolist())
                
                if i == 0:
                    tickers.extend(list(ticker))
                    dates.extend(list(date))
        
        all_probs.append(probs)
    
    results = pd.DataFrame({
        'ticker': tickers,
        'date': dates,
        'pred1': all_probs[0],
        'pred2': all_probs[1],
        'pred3': all_probs[2],
        'pred4': all_probs[3],
        'pred5': all_probs[4],
    })
    
    results['pred_mean'] = results[['pred1', 'pred2', 'pred3', 'pred4', 'pred5']].mean(axis=1)
    
    print("\n=== Results ===")
    print(results.head(10))


    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # for date in results['date'].unique():
    #     temp_df = results[results['date'] == date].drop('date', axis=1)
    #     temp_df.to_csv(f'{OUTPUT_DIR}/{date}.csv', index=False)
    # print(f"Saved to {OUTPUT_DIR}/")