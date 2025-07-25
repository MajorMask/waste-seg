import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TrashNetDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_trashnet_data(data_path):
    """
    Load TrashNet dataset and create image paths with labels
    Expected structure: data_path/class_name/image_files
    """
    image_paths = []
    labels = []
    class_names = []
    
    # Get all class directories
    for class_dir in os.listdir(data_path):
        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            class_names.append(class_dir)
    
    class_names.sort()  # Ensure consistent ordering
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    print(f"Found classes: {class_names}")
    
    # Collect all image paths and labels
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        class_idx = class_to_idx[class_name]
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    return image_paths, labels, class_names, class_to_idx

def analyze_dataset(image_paths, labels, class_names):
    """
    Perform EDA on the dataset
    """
    print("=== DATASET ANALYSIS ===")
    print(f"Total images: {len(image_paths)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Class distribution
    label_counts = Counter(labels)
    class_distribution = {class_names[idx]: count for idx, count in label_counts.items()}
    
    print("\nClass Distribution:")
    for class_name, count in class_distribution.items():
        print(f"  {class_name}: {count} images")
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.title('TrashNet Dataset - Class Distribution')
    plt.xlabel('Waste Categories')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    # Analyze image properties (sample first 100 images for speed)
    sample_indices = np.random.choice(len(image_paths), min(100, len(image_paths)), replace=False)
    widths, heights, channels = [], [], []
    
    print("\nAnalyzing image properties (sample of 100 images)...")
    for idx in sample_indices:
        try:
            img = Image.open(image_paths[idx])
            w, h = img.size
            widths.append(w)
            heights.append(h)
            channels.append(len(img.getbands()))
        except Exception as e:
            print(f"Error reading {image_paths[idx]}: {e}")
    
    print(f"\nImage Statistics:")
    print(f"  Width - Mean: {np.mean(widths):.1f}, Min: {min(widths)}, Max: {max(widths)}")
    print(f"  Height - Mean: {np.mean(heights):.1f}, Min: {min(heights)}, Max: {max(heights)}")
    print(f"  Channels - Unique values: {set(channels)}")
    
    return class_distribution

def create_data_splits(image_paths, labels, test_size=0.2, val_size=0.1):
    """
    Create train/validation/test splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\nData Splits:")
    print(f"  Train: {len(X_train)} images ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} images ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"  Test: {len(X_test)} images ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_transforms():
    """
    Define data transformations for training and validation
    """
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    Create PyTorch data loaders
    """
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = TrashNetDataset(X_train, y_train, transform=train_transform)
    val_dataset = TrashNetDataset(X_val, y_val, transform=val_transform)
    test_dataset = TrashNetDataset(X_test, y_test, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"\nData Loaders Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def visualize_samples(data_loader, class_names, num_samples=8):
    """
    Visualize sample images from the dataset
    """
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        # Denormalize
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("sample_images.png")

# MAIN EXECUTION
if __name__ == "__main__":
    # UPDATE THIS PATH TO YOUR TRASHNET DATASET LOCATION
    DATA_PATH = "./data/1/dataset-resized"  # Change this to your actual path
    
    print("Loading TrashNet dataset...")
    image_paths, labels, class_names, class_to_idx = load_trashnet_data(DATA_PATH)
    
    print("\nPerforming dataset analysis...")
    class_distribution = analyze_dataset(image_paths, labels, class_names)
    
    print("\nCreating data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(image_paths, labels)
    
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    print("\nVisualizing sample images...")
    visualize_samples(train_loader, class_names)
    
    print("\n=== SETUP COMPLETE ===")
    print("Next step: Run the model training script!")
    
    # Save important info for model training
    import json
    metadata = {
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_classes': len(class_names),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
    
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Dataset metadata saved to 'dataset_metadata.json'")
