import os
import time
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import dynamic HybridTBPredictor model
from src.models.hybrid_model import HybridTBPredictor

# Custom PyTorch Dataset
class ChestXRayDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row['image_path']
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['label'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / len(dataloader.dataset), correct / (total + 1e-8)

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-Attention Model with Dynamic Backbones")
    parser.add_argument("--backbone", type=str, default="densenet121", choices=["densenet121", "resnet50", "efficientnet_b0"], help="Select neural backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw image database folder")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to metadata.csv")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using Device: {device}")
    
    # Auto-resolve metadata path
    metadata_path = Path(args.metadata_path) if args.metadata_path else (Path("data/metadata.csv") if Path("data/metadata.csv").exists() else Path("../data/metadata.csv"))
    if not metadata_path.exists():
        print(f"[-] Error: Metadata CSV not found at {metadata_path}. Please provide a valid path via --metadata-path.")
        return

    # Auto-resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif Path("/content/colab_data").exists():
        data_dir = Path("/content/colab_data")
    elif Path("colab_data").exists():
        data_dir = Path("colab_data")
    else:
        data_dir = Path("E:/Download/TB Data/data")

    print(f"[+] Loading metadata from {metadata_path}")
    print(f"[+] Loading images from database at {data_dir}")
    
    df = pd.read_csv(metadata_path)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    
    print(f"[+] Dataset size: Train={len(train_df)}, Val={len(val_df)}")
    
    # Calculate balanced class weights
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[+] Calculated Class Weights: {class_weights}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataloaders
    train_dataset = ChestXRayDataset(train_df, data_dir, transform=train_transform)
    val_dataset = ChestXRayDataset(val_df, data_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model dynamically
    print(f"[+] Initializing HybridTBPredictor with {args.backbone} backbone...")
    model = HybridTBPredictor(backbone=args.backbone, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Save filename mapping
    weights_name = "best_tb_model.pth" if args.backbone == "densenet121" else f"best_tb_model_{args.backbone}.pth"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_save_path = models_dir / weights_name

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 4

    print(f"[+] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{args.epochs} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"[+] Saved best weights file to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[+] Early stopping triggered. Training stopped at epoch {epoch}.")
                break
                
    print("[+] Training completed successfully!")

if __name__ == "__main__":
    main()
