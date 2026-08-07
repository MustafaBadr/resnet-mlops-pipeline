"""
Training Script for ResNet-50 on CIFAR-10
==========================================

This is the heart of the project. It:
1. Loads the config (hyperparameters from YAML)
2. Downloads and prepares CIFAR-10 data
3. Runs the training loop
4. Evaluates on the test set
5. Logs params, per-epoch metrics, and the best checkpoint to MLflow
   (when config['tracker'] == 'mlflow')

TF EQUIVALENT OF THIS ENTIRE FILE:
    model = ResNet50(...)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)
    model.save('model.h5')

In PyTorch, we write each of those steps explicitly — which is exactly what
gives us the hooks to add MLflow, W&B, and other MLOps tools later.
"""

import argparse
import contextlib
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Import our model from the file we created in Step 2
from model import ResNet18


# =============================================================================
# 1. LOAD CONFIGURATION
# =============================================================================
# TF equivalent: there's no built-in config system in TF/Keras.
# Most TF projects hardcode values or use argparse. Using YAML configs is
# an MLOps best practice that works with any framework.

def load_config(config_path):
    """Load hyperparameters from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# 2. DATA PIPELINE
# =============================================================================
# This function downloads CIFAR-10 and sets up data loaders.
#
# TF EQUIVALENT:
#   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#   x_train = x_train / 255.0  # normalize
#   train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#   train_dataset = train_dataset.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
#
# PyTorch uses transforms (preprocessing steps) and DataLoaders (batching + shuffling).

def get_data_loaders(config):
    """
    Download CIFAR-10 and create train/test data loaders.
    
    A DataLoader does three things:
    1. Batches: groups images into batches of 64 (or whatever batch_size says)
    2. Shuffles: randomizes order each epoch so the model doesn't memorize sequence
    3. Loads in parallel: uses multiple CPU threads to prepare the next batch
       while the current one is training
    """
    
    # --- Transforms (preprocessing) ---
    # These run on EVERY image before it enters the model.
    
    # Training transforms include data augmentation — random flips and crops
    # that make the model more robust by showing it slightly different versions
    # of each image every epoch.
    # TF equivalent: tf.keras.layers.RandomFlip('horizontal')
    #                tf.keras.layers.RandomCrop(32, 32)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),       # 50% chance to mirror the image
        transforms.RandomCrop(32, padding=4),     # Randomly shift the image slightly
        transforms.ToTensor(),                    # Convert PIL image → PyTorch tensor
        # Also changes range from [0,255] to [0,1] and reorders to channels-first
        # TF equivalent: x / 255.0
        transforms.Normalize(                     # Normalize to zero mean, unit variance
            mean=[0.4914, 0.4822, 0.4465],        # CIFAR-10 channel means
            std=[0.2470, 0.2435, 0.2616]          # CIFAR-10 channel stds
        )
        # These magic numbers are precomputed from the CIFAR-10 training set.
        # TF equivalent: tf.keras.layers.Normalization(mean=..., variance=...)
    ])
    
    # Test transforms: NO augmentation. We want consistent evaluation.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # --- Download and load CIFAR-10 ---
    # First run downloads ~170MB. Subsequent runs use the cached copy.
    # TF equivalent: tf.keras.datasets.cifar10.load_data()
    train_dataset = datasets.CIFAR10(
        root=config['data_dir'], train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=config['data_dir'], train=False, download=True, transform=test_transform
    )

    # --- Debug mode: shrink to a tiny subset for fast smoke-testing ---
    if config.get('debug'):
        train_dataset = Subset(train_dataset, range(min(200, len(train_dataset))))
        test_dataset = Subset(test_dataset, range(min(100, len(test_dataset))))

    # --- Create DataLoaders ---
    # TF equivalent: tf.data.Dataset.batch(64).shuffle(10000).prefetch(AUTOTUNE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,                            # Randomize order each epoch
        num_workers=config['num_workers'],        # Parallel data loading
        pin_memory=True                           # Speeds up CPU→GPU transfer (no GPU for us, but good practice)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,                            # Don't shuffle test data
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples:     {len(test_dataset):,}")
    print(f"Batch size:       {config['batch_size']}")
    print(f"Training batches: {len(train_loader)}")
    
    return train_loader, test_loader


# =============================================================================
# 3. TRAINING LOOP
# =============================================================================
# This is what model.fit() does internally in TF.
# We write it explicitly so we can add MLflow logging later.

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one full pass through the training data.
    
    TF EQUIVALENT: One epoch inside model.fit().
    
    The loop for EACH BATCH:
    1. Load batch of images + labels
    2. Forward pass: model(images) → predictions
    3. Compute loss: how wrong are the predictions?
    4. Backward pass: compute gradients (which direction to adjust weights)
    5. Update weights: optimizer takes a step using the gradients
    6. Zero gradients: reset for next batch (PyTorch accumulates by default!)
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    # TF equivalent: this happens automatically during model.fit()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device (CPU for us, would be GPU in production)
        images, labels = images.to(device), labels.to(device)
        
        # --- Step 1: Zero the gradients ---
        # WHY? PyTorch accumulates gradients by default. If you don't zero them,
        # this batch's gradients get ADDED to last batch's gradients.
        # TF equivalent: TF handles this automatically inside model.fit()
        optimizer.zero_grad()
        
        # --- Step 2: Forward pass ---
        # Feed images through the model → get 10 logits per image
        # TF equivalent: model(images) or model.predict(images)
        outputs = model(images)
        
        # --- Step 3: Compute loss ---
        # CrossEntropyLoss = softmax + negative log likelihood
        # Measures how far our predictions are from the correct answers
        # TF equivalent: tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = criterion(outputs, labels)
        
        # --- Step 4: Backward pass (backpropagation) ---
        # Compute gradients: for each weight, how much does changing it reduce the loss?
        # TF equivalent: happens inside model.fit() via GradientTape
        loss.backward()
        
        # --- Step 5: Update weights ---
        # The optimizer uses gradients to adjust each weight
        # Adam: adapts learning rate per-parameter based on gradient history
        # TF equivalent: optimizer.apply_gradients() inside model.fit()
        optimizer.step()
        
        # --- Track metrics ---
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get the class with highest logit
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {running_loss/(batch_idx+1):.4f} | "
                  f"Acc: {100.*correct/total:.2f}%")
    
    # Return average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# =============================================================================
# 4. EVALUATION
# =============================================================================
# Run the model on test data WITHOUT updating weights.
#
# TF EQUIVALENT: model.evaluate(x_test, y_test)

def evaluate(model, test_loader, criterion, device):
    """Evaluate model on the test set. No gradient computation needed."""
    model.eval()  # Set to evaluation mode (disables dropout, freezes batch norm)
    # TF equivalent: happens automatically during model.evaluate()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # torch.no_grad() tells PyTorch not to track gradients during evaluation.
    # This saves memory and speeds things up — we're just measuring, not learning.
    # TF equivalent: not needed, TF handles this in model.evaluate()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


# =============================================================================
# 5. MAIN — PUTTING IT ALL TOGETHER
# =============================================================================

def main():
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(description='Train ResNet-50 on CIFAR-10')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML file')
    # Allow overriding config values from command line
    # Example: python src/train.py --epochs 10 --learning_rate 0.0001
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--tracker', type=str, default=None,
                        choices=['none', 'mlflow'],
                        help='Experiment tracker to use')
    parser.add_argument('--debug', action='store_true',
                        help='Run on a tiny data subset (200 train / 100 test) for fast smoke-testing')
    args = parser.parse_args()

    # Load config and apply any command line overrides
    config = load_config(args.config)
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.tracker is not None:
        config['tracker'] = args.tracker
    config['debug'] = args.debug
    
    # --- Device setup ---
    # Check if GPU is available. For your XPS with integrated graphics: CPU.
    # On Colab/Kaggle/RunPod: this would pick up the GPU automatically.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ResNet-18 CIFAR-10 Training")
    print(f"{'='*60}")
    print(f"Device:        {device}")
    print(f"Epochs:        {config['epochs']}")
    print(f"Batch size:    {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Optimizer:     {config['optimizer']}")
    print(f"Tracker:       {config['tracker']}")
    print(f"Debug mode:    {config['debug']}")
    print(f"{'='*60}\n")
    
    # --- Create model ---
    model = ResNet18(num_classes=config['num_classes']).to(device)
    # .to(device) moves the model to CPU or GPU
    # TF equivalent: with tf.device('/GPU:0'): model = ResNet50(...)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # --- Loss function ---
    # CrossEntropyLoss combines softmax + negative log likelihood
    # TF equivalent: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # "Sparse" because labels are integers (3), not one-hot vectors ([0,0,0,1,...])
    criterion = nn.CrossEntropyLoss()
    
    # --- Optimizer ---
    # TF equivalent: tf.keras.optimizers.Adam(learning_rate=0.001)
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                              momentum=0.9, weight_decay=5e-4)
    # NOTE: In TF, you pass the optimizer to model.compile().
    # In PyTorch, you pass the model's parameters TO the optimizer.
    # The optimizer needs to know which weights it's responsible for updating.
    
    # --- Load data ---
    train_loader, test_loader = get_data_loaders(config)

    # --- MLflow setup ---
    # Only tracks when config['tracker'] == 'mlflow'. When disabled, `run_context`
    # is a no-op so the training loop below doesn't need to branch on tracker.
    use_mlflow = config['tracker'] == 'mlflow'
    if use_mlflow:
        mlflow.set_experiment(config['experiment_name'])
        run_context = mlflow.start_run()
    else:
        run_context = contextlib.nullcontext()

    with run_context:
        if use_mlflow:
            mlflow.log_params(config)

        # --- Training loop ---
        print(f"\nStarting training...\n")
        best_acc = 0.0

        for epoch in range(1, config['epochs'] + 1):
            epoch_start = time.time()

            # Train one epoch
            # TF equivalent: one epoch inside model.fit()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Evaluate on test set
            # TF equivalent: validation_data=(x_test, y_test) inside model.fit()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch}/{config['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

            if use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                }, step=epoch)

            # Save best model
            # TF equivalent: ModelCheckpoint callback with save_best_only=True
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pt')
                # model.state_dict() = dictionary of all weight tensors
                # TF equivalent: model.save_weights('best_model.h5')
                print(f"  >> New best model saved! (Test Acc: {test_acc:.2f}%)")

        if use_mlflow:
            mlflow.log_metric('best_test_acc', best_acc)
            mlflow.log_artifact('best_model.pt')

        # --- Final summary ---
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best test accuracy: {best_acc:.2f}%")
        print(f"Model saved to: best_model.pt")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
