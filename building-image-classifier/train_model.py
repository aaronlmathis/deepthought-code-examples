import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import numpy as np
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

CONFIG = {    
    # Model architecture parameters
    'num_classes': 10,                    # CIFAR-10 classification target
    
    # Optimization hyperparameters - tuned for CIFAR-10 characteristics
    'weight_decay': 1e-4,                 # L2 regularization to prevent overfitting
    'learning_rate': 0.1,                 # Aggressive initial LR for SGD momentum
    'batch_size': 128,                    # Balance between gradient quality and memory
    'optimizer_type': 'sgd',              # SGD with momentum for residual networks
    'scheduler_type': 'multistep',        # Step decay at specific milestones
    
    # Training schedule - designed for convergence around epoch 170
    'num_epochs': 3,                    # Maximum training duration
    'early_stopping_patience': 50,       # Allow time for LR decay benefits
    'min_improvement': 0.0005,            # Threshold for meaningful progress
    
    # Learning rate decay schedule - critical for final convergence
    'scheduler_params': {
        'milestones': [150, 225],         # Decay points based on typical CIFAR-10 training
        'gamma': 0.1,                     # 10x reduction for fine-tuning
    },
    
    # SGD with momentum - proven effective for residual networks
    'optimizer_params': {
        'momentum': 0.9,                  # Accelerates convergence in relevant directions
        'nesterov': False,                # Standard momentum for stability
    },
    
    # Data paths
    'data_path': 'processed/prepared_train_dataset.pt',
    'test_size': 0.2,                     # Validation split ratio
    'random_state': 42,                   # For reproducible splits
}

class IdentityPadding(nn.Module):
    """
    Custom padding module for PyramidNet residual connections.
    Handles channel dimension mismatches and spatial downsampling in skip connections.
    Essential for maintaining gradient flow in residual blocks with changing dimensions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityPadding, self).__init__()
        # Use average pooling for spatial downsampling to preserve information
        # better than max pooling for residual connections
        if stride == 2:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.pooling = None
            
        # Calculate how many zero channels to pad for dimension matching
        self.add_channels = out_channels - in_channels
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        if self.pooling is not None:
            out = self.pooling(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)      
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                stride=1, padding=1, bias=False)    
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = IdentityPadding(in_channels, out_channels, stride)
            
        self.stride = stride

    def forward(self, x):
        shortcut = self.down_sample(x)
        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        out += shortcut
        return out
    
class PyramidNet(nn.Module):
    def __init__(self, num_layers, alpha, block, num_classes=10):
        """
        PyramidNet implementation with gradual channel widening.
        
        Key innovation: Instead of doubling channels at each stage (like ResNet),
        PyramidNet gradually increases channels throughout the network, creating
        a 'pyramid' shape that balances capacity and efficiency.
        
        Args:
            num_layers: Number of residual blocks per stage (18 = 54 total blocks)
            alpha: Total channel increase across network (48 channels added)
            block: Type of residual block to use (ResidualBlock)
            num_classes: Output classes for classification (10 for CIFAR-10)
        """
        super(PyramidNet, self).__init__()   	
        self.in_channels = 16
        
        # num_layers = 18 blocks per stage
        self.num_layers = num_layers
        self.addrate = alpha / (3*self.num_layers*1.0)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Three stages with different spatial resolutions
        # feature map size = 32x32
        self.layer1 = self.get_layers(block, stride=1)
        # feature map size = 16x16
        self.layer2 = self.get_layers(block, stride=2)
        # feature map size = 8x8
        self.layer3 = self.get_layers(block, stride=2)

        self.out_channels = int(round(self.out_channels))
        self.bn_out= nn.BatchNorm2d(self.out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(self.out_channels, num_classes)

        # Weight initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, stride):
        layers_list = []
        for _ in range(self.num_layers): 
            self.out_channels = self.in_channels + self.addrate
            layers_list.append(block(int(round(self.in_channels)), 
                                     int(round(self.out_channels)), 
                                     stride))
            self.in_channels = self.out_channels
            stride=1
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_out(x)
        x = self.relu_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x

def pyramidnet():
    """Factory function to create our PyramidNet model"""
    block = ResidualBlock
    model = PyramidNet(num_layers=18, alpha=48, block=block)
    return model

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Reduces overconfidence and improves generalization
    """
    def __init__(self, smoothing=0.1, reduction='mean', weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight
        
    def forward(self, input, target):
        """
        Args:
            input: [N, C] where N is batch size and C is number of classes
            target: [N] class indices
        """
        log_prob = F.log_softmax(input, dim=-1)
        weight = self.weight
        if weight is not None:
            weight = weight.unsqueeze(0)

        nll_loss = F.nll_loss(log_prob, target, reduction=self.reduction, weight=weight)
        smooth_loss = -log_prob.mean(dim=-1)
        
        if self.reduction == 'mean':
            smooth_loss = smooth_loss.mean()
        elif self.reduction == 'sum':
            smooth_loss = smooth_loss.sum()
            
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset with optional data augmentation"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = (image * 255).byte()
            image = self.transform(image)
        
        return image, label    

class EarlyStopping:
    """Enhanced Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, monitor='accuracy'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor.lower()
        self.counter = 0
        self.best_accuracy = 0.0
            
    def __call__(self, val_accuracy):
        """
        Check if early stopping should trigger
        
        Args:
            val_accuracy (float): Current validation accuracy
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        improved = False
        
        if val_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = val_accuracy
            improved = True
            
        if improved:
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience    

def get_scheduler(optimizer, scheduler_type, **kwargs):
    """Factory function to create learning rate schedulers"""
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'multistep':
        milestones = kwargs.get('milestones', [150, 225])
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    # Add other scheduler types as needed
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")        

def create_data_loaders(X, y, config):
    """
    Sophisticated data preparation pipeline for robust CNN training.
    
    Data augmentation strategy:
    - RandomCrop(32, padding=4): Prevents overfitting to exact pixel positions
    - RandomHorizontalFlip(): Doubles effective dataset size, improves generalization
    - Normalization with CIFAR-10 statistics: Ensures stable gradient flow
    """
    
    # Stratified split preserves class distribution in train/val sets
    X_train, X_val, y_train, y_val = train_test_split(
        X.numpy(), y.numpy(), 
        test_size=config['test_size'], 
        stratify=y.numpy(),  # Ensures proportional class representation
        random_state=config['random_state']  # Reproducible splits
    )
    
    # Convert back to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Create augmented datasets
    train_dataset = AugmentedDataset(X_train, y_train, transform=train_transform)
    val_dataset = AugmentedDataset(X_val, y_val, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, (X_train, X_val, y_train, y_val)    

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device):
    """
    Comprehensive training loop with advanced monitoring and checkpointing.
    """
    
    # Initialize tracking variables
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    learning_rates = []
    best_val_acc = 0
    best_epoch = 0
    
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], 
        min_delta=config['min_improvement'],
        monitor='accuracy'
    )

    print("=" * 80)
    print("PYRAMIDNET TRAINING STARTED")
    print("=" * 80)
    print(f"Max epochs: {config['num_epochs']}")
    print(f"Early stopping: {config['early_stopping_patience']} epochs on validation accuracy")
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Step the scheduler
        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training and validation
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        learning_rates.append(current_lr)
        
        # Print progress
        print(f"Training   → Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Validation → Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Train/Val Gap: {train_acc - val_acc:.2f}%")
        
        # Check for best model
        improved = False
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            improved = True
            
            print(f"NEW BEST! Improvement: +{improvement:.3f}% (Best: {best_val_acc:.2f}%)")
            
            # Save best model checkpoint (PyTorch 2.6+ compatible)
            os.makedirs('models', exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),  # Only save state dict
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_acc': best_val_acc,
                'train_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'learning_rates': learning_rates
                },
                # Add metadata for compatibility
                'model_config': {
                    'num_layers': 18,
                    'alpha': 48,
                    'num_classes': config['num_classes']
                }
            }
            
            # Save with explicit format specification
            torch.save(checkpoint, 'models/best_pyramidnet_model.pth')
            print(f"Model checkpoint saved: models/best_pyramidnet_model.pth")
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\nEARLY STOPPING TRIGGERED!")
            print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
            print(f"Training stopped at epoch {epoch + 1}")
            break
        
        # Milestone summaries
        if (epoch + 1) % 10 == 0:
            print(f"\nMILESTONE SUMMARY (Epoch {epoch + 1}):")
            print(f"   Best Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
            print(f"   Current Gap: {train_acc - val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

def evaluate_model(model, val_loader, device):
    """Comprehensive model evaluation with detailed performance metrics"""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # Extract both predictions and confidence scores
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    overall_accuracy = np.mean(all_preds == all_targets)
    
    # Print results
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Average Precision: {np.mean(precision):.4f}")
    print(f"Average Recall: {np.mean(recall):.4f}")
    print(f"Average F1-Score: {np.mean(f1):.4f}")
    
    return all_preds, all_targets, all_probs

def plot_training_curves(training_results):
    """Plot comprehensive training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(training_results['train_losses']) + 1)
    
    # Loss curves
    ax1.plot(epochs, training_results['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_results['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, training_results['train_accs'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, training_results['val_accs'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axhline(y=training_results['best_val_acc'], color='g', linestyle='--', 
                label=f'Best Val Acc: {training_results["best_val_acc"]:.2f}%')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3.plot(epochs, training_results['learning_rates'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Overfitting analysis
    gap = np.array(training_results['train_accs']) - np.array(training_results['val_accs'])
    ax4.plot(epochs, gap, 'purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.fill_between(epochs, gap, 0, alpha=0.3, color='purple')
    ax4.set_title('Overfitting Analysis (Train-Val Gap)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Gap (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(targets, predictions, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Setup logging and device
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading data from: {CONFIG['data_path']}")
    X, y = torch.load(CONFIG['data_path'])
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(torch.unique(y))}")

    # Create data loaders
    train_loader, val_loader, data_splits = create_data_loaders(X, y, CONFIG)

    # Create PyramidNet model
    model = pyramidnet().to(device)

    # Model summary
    print("\n" + "=" * 80)
    print("PYRAMIDNET MODEL SUMMARY")
    print("=" * 80)
    print("Architecture: PyramidNet with ResidualBlocks")
    print("Design Philosophy: Gradual channel widening vs. ResNet's doubling")
    print("num_layers: 18 blocks per stage (54 total residual blocks)")
    print("alpha: 48 (total channel increase from 16 to ~64)")
    print("num_classes: 10 (CIFAR-10 classification)")

    try:
        summary(model, (3, 32, 32))
    except:
        print("torchsummary not available, skipping detailed summary")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Setup training components
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['optimizer_params']['momentum'],
        weight_decay=CONFIG['weight_decay'],
        nesterov=CONFIG['optimizer_params']['nesterov']
    )

    scheduler = get_scheduler(
        optimizer, 
        CONFIG['scheduler_type'], 
        **CONFIG['scheduler_params']
    )

    # Run training
    training_results = run_training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        CONFIG, device
    )

    # Final evaluation
    all_preds, all_targets, all_probs = evaluate_model(model, val_loader, device)

    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    
    plot_training_curves(training_results)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(all_targets, all_preds, class_names)

    # Training efficiency analysis
    total_epochs = len(training_results['train_losses'])
    convergence_epoch = training_results['best_epoch']

    print(f"\nTRAINING EFFICIENCY:")
    print(f"Total Epochs: {total_epochs}")
    print(f"Convergence Epoch: {convergence_epoch}")
    print(f"Best Validation Accuracy: {training_results['best_val_acc']:.2f}%")

    if total_epochs > convergence_epoch:
        wasted_epochs = total_epochs - convergence_epoch
        efficiency = 100 * (1 - wasted_epochs / total_epochs)
        print(f"Training Efficiency: {efficiency:.1f}% (saved {wasted_epochs} epochs)")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()