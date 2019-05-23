import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
def create_train_dataset(batch_size = 128, root = '../data'):

    transform_train = transforms.Compose([
     transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader
def create_test_dataset(batch_size = 128, root = '../data'):
    transform_test = transforms.Compose([
     transforms.ToTensor(),
    ])
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader


if __name__ == '__main__':
    print(create_train_dataset())
    print(create_test_dataset())

