import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
num_classes = 10


# 1-1 Create a neural network
class NNetwork(nn.Module):
    def __init__(self):
        super(NNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16)  
        self.fc2 = nn.Linear(16, 16)        
        self.fc3 = nn.Linear(16, num_classes)      

    def forward(self, x):
        x = x.view(-1, 28 * 28)         
        x = torch.relu(self.fc1(x))     
        x = torch.relu(self.fc2(x))     
        x = self.fc3(x) # No softmax because we use CrossEntropyLoss
        # When using the softmax activation along with the cross-entropy loss 
        # (which is LogSoftmax + NLLLoss combined into one function), 
        # the two functions effectively cancel each other out, leading to incorrect 
        # gradients and learning behavior.     
        return x
    
model = NNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # without momentum to show learning curve. You get better performance with momentum

train_losses = []
val_losses = []

num_epochs = 20
for epoch in range(num_epochs):
    model.train() 
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    train_losses.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # 1b Report your accuracy, is this satisfactory? 
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
    print(f"Test Accuracy: {accuracy:.2f}%")
    val_losses.append(loss.item())

torch.save(model.state_dict(), 'model.pth')

# 1c Plot the loss curve.
plt.figure(figsize=(8, 6))
plt.plot(range(num_epochs), train_losses, label="Training loss")
plt.plot(range(num_epochs), val_losses, label="Validation loss")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('graph.png')
plt.show()
