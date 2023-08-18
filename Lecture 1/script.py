import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

# henter MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

'''
1. create a neural network:
    a. initialize 3 layers
'''    

class ThreeLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input tensor if needed
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   

'''    
    b. define the forward function:
        - reshape the data to a fully connected layer. Hint: use .view()
        - let the input pass through the different layers
        - consider what activation function you want to use in between the layers, and for the final layer
    c. loss function and optimizer:
        - consider what loss function and optimizer you want to use in between the layers, and for the final layer
    d. create the training loop
    e. create the evaluation loop
    f. save the model
2. report you accuracy, is this satisfactory? why/why not?
3. plot the loss curve

'''