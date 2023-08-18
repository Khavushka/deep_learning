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

# Exercise 1
class NNetwork(nn.Module):
    def __init__(self):
        super(NNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16)  
        self.fc2 = nn.Linear(16, 16)          
        self.fc3 = nn.Linear(16, num_classes)      

    def forward(self, x):
        x = x.flatten(start_dim=1)        
        x = torch.relu(self.fc1(x))     
        x = torch.relu(self.fc2(x))      
        x = self.fc3(x) # No softmax because we use CrossEntropyLoss
        # When using the softmax activation along with the cross-entropy loss 
        # (which is LogSoftmax + NLLLoss combined into one function), 
        # the two functions effectively cancel each other out, leading to incorrect 
        # gradients and learning behavior. 
        return x

    def apply_regularization(self, layers, regularization):
        regularization_loss = 0  
        for layer in layers:
            if hasattr(layer, 'weight'):
                if regularization == "L1":
                    p = 1
                    L_lambda = 0.001
                elif regularization == "L2":
                    p = 2
                    L_lambda = 0.005
                else:
                    return 0
                L_reg_layer = L_lambda * torch.norm(layer.weight, p=p)
                regularization_loss += L_reg_layer
            return regularization_loss
            
def train(model, regularization: str = 'None', num_epochs: int = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    train_losses = []
    val_losses = []
    accuracy_list = []
    
    for epoch in range(num_epochs):
        model.train() 
        train_loss_list = []
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            
            loss = criterion(outputs, targets)   
            train_loss = loss     
            loss += model.apply_regularization(layers=[model.fc2, model.fc3], regularization=regularization)    
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(train_loss.item())
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_losses.append(train_loss)
        accuracy, val_loss = test(model=model, criterion=criterion)
        accuracy_list.append(accuracy)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")
    return train_losses, val_losses, accuracy_list
def test(model, criterion):
    model.eval()  
    correct = 0
    total = 0
    val_losses = []
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            val_losses.append(loss.item())
    val_loss = sum(val_losses) / len(val_losses)
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy, val_loss

reg_cases = ["None", "L1", "L2"]
for reg in reg_cases:
    print(reg + " regularization")
    model = NNetwork()
    train_loss, val_loss, accuracy = train(model=model, regularization=reg, num_epochs=10)

    torch.save(model.state_dict(), "3-1" + reg + ".pth")
    
    # Plotting
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("3-1_" + reg + "_plot.png")




    