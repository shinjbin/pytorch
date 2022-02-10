import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device <{device}>')

# hyperparameters
learning_rate = 0.001
epochs = 3
batch_size = 4
momentum = 0.9

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=transform, train=False, download=True)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def imgshow(loader):
    dataiter = iter(loader)
    image, label = dataiter.next()
    plt.imshow(torchvision.utils.make_grid(image, normalize=True).permute(1,2,0))
    plt.show()


# imgshow(train_loader)

# define neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


neuralnet = NeuralNet().to(device)

# loss
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(neuralnet.parameters(), lr=learning_rate, momentum=momentum)

# for epoch in range(epochs):
#     running_loss = 0.0
#     # train
#     for batch, data in enumerate(train_loader, 0):
#         image, label = data[0].to(device), data[1].to(device)
#
#         pred = neuralnet(image)
#         loss = criterion(pred, label)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if batch % 2000 == 1999:
#             print(f'[{epoch+1}, {batch+1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0
#
#     # test
#     size = len(test_loader.dataset)
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for image, label in test_loader:
#             image = image.to(device)
#             label = label.to(device)
#             pred = neuralnet(image)
#             test_loss += criterion(pred, label).item()
#             correct += (pred.argmax(1) == label).type(torch.float).sum().item()
#     test_loss /= batch_size
#     correct /= size
#
#     print(f'\n Accuracy: {(100*correct):.1f}%, Avg loss: {test_loss:.8f} \n-------------------------')


print('finish')

PATH = './MNIST_CLASSIFIER.pth'
# torch.save(neuralnet.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = dataiter.next()
plt.imshow(torchvision.utils.make_grid(images).permute(1,2,0), vmin=0, vmax=255)
plt.show()

# ground truth
print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(4)))

# prediction
neuralnet = NeuralNet()
neuralnet.load_state_dict(torch.load(PATH))

outputs = neuralnet(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(4)))
