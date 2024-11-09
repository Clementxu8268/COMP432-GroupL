import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import numpy as np
import joblib  # For saving/loading the encoder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset and split into train and test sets
dataset = datasets.ImageFolder("/Users/alan/Desktop/Fall_2024/COMP 432/Project/Dataset 1/Modified", transform=transform)

# Split indices for train and test datasets
train_indices, test_indices = train_test_split(
    list(range(len(dataset))), test_size=0.3, stratify=dataset.targets, random_state=42
)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the ResNet model
model = models.resnet18(weights=None)  # Set weights=None for training from scratch
num_classes = 3  # Adjust for your classes: MUS, NORM, STR
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 8  # Adjust this for more thorough training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save the trained encoder (without the classification layer) for Task 2
encoder = nn.Sequential(*list(model.children())[:-1])  # Extracts model up to the last layer
joblib.dump(encoder, 'trained_resnet_encoder.pkl')

# Evaluation on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Feature extraction and visualization for Task 1
encoder.eval()
features, labels = [], []
with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        output = encoder(images).squeeze()  # (batch_size, feature_size)
        features.extend(output.cpu().numpy())
        labels.extend(lbls.numpy())

# Convert features and labels to NumPy arrays
features = np.array(features)  # Convert list to NumPy array
labels = np.array(labels)      # Convert labels list to NumPy array

# t-SNE on extracted features
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1, 2], label='Class')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE of Extracted Features from ResNet Encoder on Test Set")
plt.show()
