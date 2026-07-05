import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import GalaxyClassifier

# Load preprocessed data
print("Loading data...")
X_train = np.load("data/X_train.npy")
X_val = np.load("data/X_val.npy")
y_train = np.load("data/y_train.npy")
y_val = np.load("data/y_val.npy")

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = GalaxyClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
NUM_EPOCHS = 10
best_val_accuracy = 0

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels, in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100 * correct / total

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


    val_accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.3f} | "
          f"Train Acc: {train_accuracy:.1f}% | "
          f"Val Acc: {val_accuracy:.1f}%")
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "models/best_model.pt")
        print(f" -> New best model saved! ({val_accuracy:.1f}%)")

print(f"\nTraining complete! Best validation accuracy: {best_val_accuracy:.1f}%")