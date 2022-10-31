import pickle
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import flowers102, dtd
from torchvision import models
from numpy.linalg import norm
import time
import random




class ImageNet(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        #target = np.zeros(1000)
        label = self.labels[idx]
        #target[label] = 1.
        #return (self.transform(img), target)
        return (self.transform(img), label)

    
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def get_train_data():
    X = []
    Y = []

    for i in range(1, 11):
        out = unpickle('../cntk_imagenet/Imagenet32_train/train_data_batch_' + str(i))
        x = out['data']
        mean_image = out['mean']
        print(x.dtype)

        y = out['labels']
        y = [i-1 for i in y]
        data_size = x.shape[0]

        img_size = 32
        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2],
                       x[:, img_size2:2*img_size2],
                       x[:, 2*img_size2:]))

        x = x.reshape((x.shape[0], img_size, img_size, 3))
        x = np.rollaxis(x, -1, 1)

        X.append(x)
        Y.extend(y)

    X = np.concatenate(X, axis=0)

    return X, np.array(Y)

def get_test_data():
    out = unpickle('../cntk_imagenet/Imagenet32_val/val_data')
    x = out['data']
    y = out['labels']

    y = [i-1 for i in y]
    data_size = x.shape[0]

    img_size = 32
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2],
                   x[:, img_size2:2*img_size2],
                   x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    x = np.rollaxis(x, -1, 1)

    return x, np.array(y)



def group_by_class(X_train, y_train):
    num_classes = np.sum(y_train, axis=0)
    print(num_classes)
    class_lookup = {}
    labels = y_train.nonzero()[1]

    for idx, label in enumerate(labels):
        x = X_train[idx]
        if label in class_lookup:
            class_lookup[label].append(x)
        else:
            class_lookup[label] = [x]
    return class_lookup


def remap_classes(class_list):
    count = 0
    label_map = {}
    for l in class_list:
        label_map[l] = count
        count += 1
    return label_map


def select_classes(class_lookup, class_list):
    X_train = []
    y_train = []

    label_map = remap_classes(class_list)

    for l in class_list:
        X = class_lookup[l]
        X_train.extend(X)
        ys = np.zeros((len(X), len(class_list)))
        ys[:, label_map[l]] = 1.
        y_train.extend(ys)

    X_train = np.array(X_train).astype('float32')
    y_train = np.array(y_train).astype('float32')
    return X_train, y_train


def get_imagenet_data(classlist=None):
    X_train, y_train = get_train_data() 
    X_test, y_test = get_test_data() 

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    if classlist is not None:
        class_lookup = group_by_class(X_train, y_train)
        X_train, y_train = select_classes(class_lookup, classlist)

    print("Train set: ", X_train.shape, y_train.shape)
    print("Test set: ", X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test)



def get_imagenet_dataloaders():
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()

    X_train = torch.from_numpy(X_train) / 255.
    y_train = torch.from_numpy(y_train) 
    X_test = torch.from_numpy(X_test) / 255.
    y_test = torch.from_numpy(y_test)
    
    trainset = ImagenetDataset(X_train, y_train)
    testset = ImagenetDataset(X_test, y_test)
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1024, shuffle=False)

    return train_loader, test_loader


class ImagenetDataset(Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return (img, label)

    
def get_cifar_data():
    loader = transforms.Compose([
        transforms.ToTensor(),        
    ])

    cifar_path = '~/NeuralNetworks/expressivity_through_depth/cifar_experiments/data/'
        
    training_data = torchvision.datasets.CIFAR10(
        root=cifar_path,
        train=True,
        download=False,
        transform=loader
    )
    
    test_data = torchvision.datasets.CIFAR10(
        root=cifar_path,
        train=False,
        download=False,
        transform=loader
    )
    
    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    return train_loader, test_loader


def get_flower102_data():
    loader = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
    ])

    training_data = flowers102.Flowers102(
        root='../laplace_pretrained_resnet/flower102_data',
        split='train',
        #train=True,
        download=False,
        transform=loader
    )
    
    test_data = flowers102.Flowers102(
        root='../laplace_pretrained_resnet/flower102_data/',
        split='test',
        #train=False,
        download=False,
        transform=loader
    )
    
    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    return train_loader, test_loader

    
def get_svhn_data():
    loader = transforms.Compose([
        transforms.ToTensor(),        
    ])

    training_data = torchvision.datasets.SVHN(
        root='../laplace_pretrained_resnet/svhn_data',
        split='train',
        #train=True,
        download=False,
        transform=loader
    )
    
    test_data = torchvision.datasets.SVHN(
        root='../laplace_pretrained_resnet/svhn_data/',
        split='test',
        #train=False,
        download=False,
        transform=loader
    )

    train_loader = DataLoader(training_data, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    return train_loader, test_loader
    

def get_dtd_data():
    loader = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
    ])

    training_data = dtd.DTD(
        root='../laplace_pretrained_resnet/dtd_data',
        split='train',
        #train=True,
        download=True,
        transform=loader
    )
    
    test_data = dtd.DTD(
        root='../laplace_pretrained_resnet/dtd_data',
        split='test',
        #train=False,
        download=True,
        transform=loader
    )
    
    train_loader = DataLoader(training_data, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    return train_loader, test_loader

