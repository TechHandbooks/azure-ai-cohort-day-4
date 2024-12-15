import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Load and Transform Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download FashionMNIST dataset
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

# 2. Visualize Sample Data
def visualize_sample(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Label mapping for FashionMNIST
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Visualize first image and its label
    plt.figure(figsize=(3, 3))
    plt.imshow(images[0].numpy().squeeze(), cmap='gray')
    plt.title(f"Label: {classes[labels[0]]}")
    plt.axis('off')
    plt.show()

# Visualize
visualize_sample(trainloader)

# 3. Define a Simple Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = SimpleNN()

# 4. Define Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Train the Model
epochs = 3
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

# 6. Test the Model
dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to('cpu'), labels.to('cpu')
output = model(images)
_, preds = torch.max(output, 1)

# Print Predictions for First 5 Images
for i in range(5):
    print(f"Image {i+1}: Actual: {labels[i].item()}, Predicted: {preds[i].item()}")
