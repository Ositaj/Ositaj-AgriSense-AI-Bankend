import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 🔧 Paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# 📐 Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 📂 Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ⚙️ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 🧠 Load pre-trained model (ResNet18)
model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)

# 🔁 Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ⚙️ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔁 Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"📚 Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# 💾 Save model
torch.save(model.state_dict(), 'plant_disease_model.pth')
print("✅ Model trained and saved as 'plant_disease_model.pth'")
