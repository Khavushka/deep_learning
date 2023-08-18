import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Download and load the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3) # 26
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)# 13
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3)# 11
        self.conv3 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3)# 11
        self.conv4 = nn.Conv2d(in_channels=56, out_channels=28, kernel_size=3)# 9
        self.fc1 = nn.Linear(28 * 9 * 9, 64) # input_features = out_channels * height * width
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x1 = torch.relu(self.conv2(x))
        x2 = torch.relu(self.conv3(x))
        x = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.conv4(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, num_epochs: int = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    train_losses = []
    val_losses = []
    accuracy_list = []
    for epoch in range(num_epochs):
        model.train()

        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        accuracy, val_loss = test(model, criterion)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        accuracy_list.append(accuracy)
        print('Epoch: {} / {}  Validation Loss: {}  Accuracy: {} %'.format(epoch+1, num_epochs, val_loss.item(), accuracy))
    return train_losses, val_losses, num_epochs, accuracy_list


def test(model, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy, loss

model = CNN()  
train_loss, val_loss, num_epochs, accuracy_list = train(model, 20)
torch.save(model.state_dict(), 'model5-1.pth')

# visualization loss 
sns.set_style('whitegrid')
plt.plot(range(1, num_epochs+1), train_loss, color='blue', linestyle='-', linewidth=2, label='Training loss')
plt.plot(range(1, num_epochs+1), val_loss, color='red', linestyle='-', linewidth=2, label='Validation loss')
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of epochs")
plt.legend()
plt.savefig('5-1loss_curve.png')
plt.show()

# visualization accuracy 
sns.set_style('whitegrid')
plt.plot(range(1, num_epochs+1),accuracy_list,color = "red", linestyle='-', linewidth=2)
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of epochs")
plt.savefig('5-1accuracy_curve.png')
plt.show()