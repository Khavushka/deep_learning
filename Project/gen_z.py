# Import packages
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

input_dim = (512, 512)
class gen_z(nn.Module):
    def __init__(self):
        super(gen_z, self).__init__()

        # Convolutional network
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, padding=5//2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 256px out

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=3//2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4)  # 64 out

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=3//2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # 32px out

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=3//2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=8)  # 4 out

        # Fully connected network
        self.fc1 = nn.Linear(1024, 500)  # 40 channels * 3px * 3px
        self.fc2 = nn.Linear(500, 250)  # 40 channels * 3px * 3px
        self.fc3 = nn.Linear(250, 100)  # 40 channels * 3px * 3px
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        # Convolutional network
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.maxpool4(x)

        # Fully connected network
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return (x)
    
    # Define function to train network
def train_NN(network, num_epochs, learning_rate=0.01, momentum=0.9, L1=False, L2=False, my_lambda=0.01):
    train_loss = []
    vall_loss = []
    accuracy_deci = []
    current_max_accuracy = 0
    criterion = nn.CrossEntropyLoss() # We chose Cross-Entropy as our loss function.
    # Stochastic Gradient Descent is used as optimization function.
    # We've also tried Adam and Adagrad, but found that SGD gave the best results.
    optimizer = torch.optim.SGD(
        network.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        network.train()
        running_loss = 0
        # Iterates through the training loader data.
        for images, labels in train_loader:
            images = images.to(device)          # Sends images to GPU
            labels = labels.to(device)          # Sends labels to GPU
            optimizer.zero_grad()               # Clear gradients
            outputs = network(images)           # Forward pass
            loss = criterion(outputs, labels)   # Compute loss
            if L1 == True:                      # L1 regularization if True
                loss += my_lambda * torch.norm(network.fc1.weight, p=1)
            if L2 == True:                      # L2 regularization if True
                loss += my_lambda * torch.norm(network.fc1.weight, p=2)
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update weights
            running_loss += loss.item()

        # Calculate average training loss pr. epoch.
        # To normalise the loss curve across different batches to avoid spiikes.
        train_loss.append(running_loss/len(train_loader))
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {(running_loss/len(train_loader)):.4f}')

        # Set the model to evaluation mode
        network.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            # Iterates through validation dataset.
            for images, labels in vali_loader:
                # Sends images to GPU
                images = images.to(device)
                # Sends labels to GPU
                labels = labels.to(device)
                outputs = network(images)
                # Return predicted labels
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total              # Compute accuracy
            loss = criterion(outputs, labels)       # Compute loss
            print(f'Test Accuracy: {accuracy:.2%}')

        # Saves the model if it has the highest accuracy so far
        if accuracy > current_max_accuracy:
            torch.save(net.state_dict(), 'group_4.pth')
            current_max_accuracy = accuracy

        # Append values for graph
        accuracy_deci.append(accuracy)
        vall_loss.append(loss.item())

    # Plots training loss, validation loss and accuracy pr. training session.
    epochs = list(range(1, len(train_loss)+1, 1))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, vall_loss, label="Validation loss")
    plt.plot(epochs, accuracy_deci, label="Accuracy")
    plt.title("Network development")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.text(x=len(accuracy_deci)-1/10*len(accuracy_deci), y=0.9,
             s=f'Max: {max(accuracy_deci):.4}')
    plt.ylim([0, 1])
    plt.xlim(left=1)
    plt.legend(loc='center left')
    plt.savefig(fname='group 4 graph')
    plt.axis('off')
    plt.show()
    plt.clf()



# Define function to evaluate network performance.
def evaluate(model, test_loader):
    # Set model to evaluation mode. I.e. no weights will be changed.
    model.eval()
    correct = 0
    total = 0
    misclassified_images = []

    with torch.no_grad():
        # Iterates through test_loader batches.
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)                  # Sends images to GPU.
            labels = labels.to(device)                  # Sends labels to GPU.
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)   # Returns predicted class.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append misclassified images to list for plotting.
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified_images.append(
                        (inputs[i].cpu(), predicted[i], labels[i]))

    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')

    # Plot misclassified images.
    num_images_to_show = min(25, len(misclassified_images))
    plt.figure(figsize=(16, 16))
    for idx in range(num_images_to_show):
        img, predicted_label, true_label = misclassified_images[idx]

        img = (img.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        img = 1-img.squeeze()

        plt.subplot(5, 5, idx + 1)
        plt.imshow(img, cmap='Greys', vmin=0, vmax=1)
        plt.title(f'P: {predicted_label} / T: {true_label}', fontsize=6)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname='group_4_misclassified_immages')
    plt.show()



# Split data into 6 folders:
picture_input('data/data')

# Transformations for training data:
transform_train = transforms.Compose([
    transforms.Resize((512, 512), antialias=None),  # Resize to 512x512
    transforms.Grayscale(1),                        # Transform 3 color channels to grayscale (1channel)
    transforms.ColorJitter(contrast=0.1),           # Give random contrast to images
    transforms.ToTensor(),                          # Transform images to tensors
    transforms.Normalize((0.5,), (0.5,)),           # Normalize image values to mean 0.5 and standard deviation 0.5
    transforms.RandomResizedCrop(size=512, scale=(
        0.75, 1), ratio=(1, 1), antialias=None)     # Randomly crop part of images to force network not to train on full image.
])

# Transformation for test and validation data. Only keep transformation to fit the network input.
transform_test_vali = transforms.Compose([
    transforms.Resize((512, 512), antialias=None),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# Create datasets for each stage.
# Imagefolder looks at eg. the training folder and adds the subfolders as a class each.
train_dataset = ImageFolder(
    root='data/data/training', transform=transform_train)
vali_dataset = ImageFolder(root='data/data/validation',
                           transform=transform_test_vali)
test_dataset = ImageFolder(root='data/data/testing',
                           transform=transform_test_vali)

# Create dataloaders for each stage.
# Shuffle the training data to ensure non patterned sequences of images.
train_loader = DataLoader(dataset=train_dataset, batch_size=16,
                          shuffle=True, num_workers=0, pin_memory=True)
vali_loader = DataLoader(dataset=vali_dataset, batch_size=16,
                         shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16,
                         shuffle=False, num_workers=0, pin_memory=True)

# Define network as variable and transfer to GPU.
net = group_4()
device = torch.device('cuda:0')
net.to(device)


# Train the network
train_NN(network=net,
         L2=False,              # We don't use ANY regularization as we don't see any overfitting in the model.
         num_epochs=2,          # Select number of epoch to run.
         learning_rate=0.001,)  # We testet several learning rates and found that 0.001 gave a good

# Load the best network
net.load_state_dict(torch.load('group_4.pth', map_location=device))

# Show misclassified images
evaluate(net, test_loader)