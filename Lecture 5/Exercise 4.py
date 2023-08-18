import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Download and load the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Exercise 1: Dataset Preprocessing

print("Training dataset shape:", train_dataset.data.shape)
print("Test dataset shape:", test_dataset.data.shape)

    #   a. Output the dimensions of the images
image, label = train_dataset[0]
image_height, image_width = image.size(1), image.size(2)
print("Image dimensions:", image_height, "x", image_width)


    #   b. Plot the number distribution using a bar histogram

def histogram():
    label_list = np.empty(10)
    for label in train_dataset.targets:  # Use train_dataset.targets to access labels
        label_list[label.item()] += 1

    digits = [str(i) for i in range(10)]

    plt.bar(digits, label_list)
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title('Histogram of MNIST Digit Distribution')
    plt.savefig('outputs/histogram.png')


# Exercise 2: Design a CNN

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3)  # 26x26
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3)  # 24x24
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 12x12
        self.conv3 = nn.Conv2d(in_channels=28, out_channels=64, kernel_size=3)  # 10x10
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 5x5
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)  # 3x3
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 10)



    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train(model, num_epochs: int = 10):
    torch.save(model.state_dict(), 'model_pre_train_state.pth')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    model.to(device=device)
    losses = []

    for epoch in range(num_epochs):
        model.train()

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Exercise 3: Check Results

    #   a.	Plot the first convolutional layer filters before and after training the model.
def save_filters(model, filter_file):
    plt.figure(figsize=(10, 6))
    for i, filter in enumerate(model.conv1.weight):
        plt.subplot(4, 7, i + 1)
        plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
        plt.axis('off')
        plt.savefig(filter_file)

    #   b.	Feed the first convolutional layer an image and plot it’s feature map.
def save_first_feature_map(model, row_size, col_size):
    img_batch = next(iter(test_loader))[0].to(device)
    conv1_output = model.conv1(img_batch[0])
    layer_visualization = conv1_output.data
    for i, feature_map in enumerate(layer_visualization):
        plt.subplot(row_size, col_size, i + 1)
        plt.imshow(feature_map.cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.savefig('outputs/feature_maps.png')

    #   c.	Plot some incorrectly labelled data points to see if it makes sense it was wrongly classified.
def get_mislabeled(model):
    mislabeled_data = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            incorrect_mask = pred != target

            for i in range(len(data)):
                if incorrect_mask[i]:
                    image_data = data[i].cpu().numpy()
                    true_label = target[i].cpu().item()
                    predicted_label = pred[i].cpu().item()

                    mislabeled_data.append({
                        'image': image_data,
                        'true_label': true_label,
                        'predicted_label': predicted_label
                    })

                    if len(mislabeled_data) >= 10:
                        break

            if len(mislabeled_data) >= 10:
                break
    return mislabeled_data


def plot_mislabeled(mislabeled_data):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.5)

    for i, data_entry in enumerate(mislabeled_data):
        row = i // 5
        col = i % 5
        ax = axes[row, col]

        image = data_entry['image']
        true_label = data_entry['true_label']
        predicted_label = data_entry['predicted_label']

        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"True: {true_label}, Predicted: {predicted_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/mislabeled.png')


model = CNN()
pre_trained_model = CNN()

model.to(device=device)
pre_trained_model.to(device=device)

train_model = False
if train_model:
    train(model=model, num_epochs=10)
    test(model=model)
    torch.save(model.state_dict(), 'model_post_train_state.pth')
else:
    model.load_state_dict(torch.load('model_post_train_state.pth'))

histogram()
save_filters(pre_trained_model, filter_file='outputs/pre_train_filters.png')
save_filters(model, filter_file='outputs/post_train_filters.png')
save_first_feature_map(model,4,7)

mislabel_data = get_mislabeled(model)
plot_mislabeled(mislabel_data)