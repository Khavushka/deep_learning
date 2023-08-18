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
1. Regularization:
    - add L1 regularization to the 2nd layer (the layer after input layer)
    - add L2 regularization instead on the 2nd layer
    - what do you observe? (Hint: the lambda value used has a big impact on performance.)
    - what is the purpose of adding regularization?
'''

class MyNetwork(nn.Models):
    def __init__(self, input_size, output_size):
        super(NNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, input_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x))
        return x
    
net = MyNetwork(input_size = 10, output_size = 10)

loss = criterion(outputs, targets)
loss += L2_lambda * torch.norm(net.fc1.weight, p = 2)

'''
2. dropout:
    - add a dropout layer between the first and second layer. what do you observe?
    - what is the purpose of adding dropout?
'''
torch.nn.Dropout(p-0.5, inplace = False)

class NNetwork(nn.Module):
    def __init__(self):
        super(NNetwork, self).__init__()
        self.dropout == nn.Dropout(0.2)
        
    def forward(self, x):
        #drop is used in between two layers
        x = self.dropout(x)


'''
3. layers:
    - experiment with different amount of layers. what do you observe?
    - experiment with different depths of layers. what do you observe?
'''



'''
4. momentum:
    - try to add momentum to the SGD optimizer.
    - test different values of momentum. what value do you get the highest accuracy?
    - what happens if momentum is too high?
'''