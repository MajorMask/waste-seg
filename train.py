import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import json
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os

# Import data loading functions from eda.py
from eda import load_trashnet_data, create_data_splits, create_data_loaders

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6, model_name='efficientnet'):
        super(WasteClassifier, self).__init__()
        
        if model_name == 'efficientnet':
            # Use EfficientNet-B0 as backbone
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # Replace classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        elif model_name == 'resnet':
            # Use ResNet50 as backbone
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            # Replace final layer
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=10, device='cuda'):
    """
    Train the model with validation
    """
    since = time.time()
    
    # Keep track of best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            pbar = tqdm(dataloader, desc=f'{phase.upper()}')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{torch.sum(preds == labels.data).double() / inputs.size(0):.4f}'
                })
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.cpu().numpy())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.cpu().numpy())
                
                # Update learning rate scheduler
                scheduler.step()
                
                # Deep copy the model if it's the best so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_acc': best_acc.cpu().numpy()
    }
    
    return model, history

def evaluate_model(model, test_loader, class_names, device='cuda'):
    """
    Evaluate model on test set and generate detailed metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy, all_preds, all_labels, all_probs

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['val_accs'], label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def save_model(model, history, metadata, model_path='waste_classifier.pth'):
    """
    Save the trained model and metadata
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'metadata': metadata,
        'model_architecture': 'efficientnet_b0'
    }, model_path)
    print(f"Model saved to {model_path}")

def load_or_create_data_loaders(data_path, batch_size=32):
    """
    Load data loaders, creating them if metadata doesn't exist
    """
    metadata_path = 'dataset_metadata.json'
    
    if os.path.exists(metadata_path):
        print("Loading existing dataset metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print("Dataset metadata not found. Creating data loaders from scratch...")
        # Load data
        image_paths, labels, class_names, class_to_idx = load_trashnet_data(data_path)
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(image_paths, labels)
        
        # Save metadata
        metadata = {
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'num_classes': len(class_names),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Dataset metadata saved.")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test, batch_size
        )
        
        return train_loader, val_loader, test_loader, metadata
    
    # If metadata exists, recreate data loaders
    print("Recreating data loaders from existing metadata...")
    image_paths, labels, class_names, class_to_idx = load_trashnet_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(image_paths, labels)
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )
    
    return train_loader, val_loader, test_loader, metadata

# MAIN TRAINING SCRIPT
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    DATA_PATH = "./data/1/dataset-resized"  # Update this path to your dataset
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Check if data path exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path '{DATA_PATH}' does not exist!")
        print("Please update the DATA_PATH variable to point to your TrashNet dataset.")
        exit(1)
    
    print("Loading/creating data loaders...")
    try:
        train_loader, val_loader, test_loader, metadata = load_or_create_data_loaders(
            DATA_PATH, BATCH_SIZE
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure your dataset is in the correct format and path is correct.")
        exit(1)
    
    num_classes = metadata['num_classes']
    class_names = metadata['class_names']
    
    print(f"Training model for {num_classes} classes: {class_names}")
    
    # Initialize model
    model = WasteClassifier(num_classes=num_classes, model_name='efficientnet')
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model
    print("Starting training...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_accuracy, preds, labels, probs = evaluate_model(model, test_loader, class_names, device)
    
    # Save model
    save_model(model, history, metadata)
    
    print("Training completed successfully!")
    print(f"Final test accuracy: {test_accuracy:.4f}")