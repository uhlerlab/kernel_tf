import dataset
import neural_trainer
import neural_model as nm
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pickle
from numpy.linalg import norm


NUM_CLASSES = 10

def get_samples(net, loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        print(idx, len(loader))
        inputs, labels = batch
        inputs = inputs.cuda()
        out = net(inputs).view(len(inputs), -1).cpu().data.numpy()
        targets = np.zeros((len(labels), NUM_CLASSES))
        targets[np.arange(len(labels)), labels] = 1.
        X.append(out)
        y.append(targets)
        #if idx > 30:
        #    break
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def main():
    SEED = 17
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    net = nm.Net()
    fname = 'trained__model.pth'
    
    d = torch.load('./saved_models/' + fname)
    
    net.load_state_dict(d['state_dict'])

    # Can use dataset loaders to load cifar, svhn, or dtd data
    trainloader, testloader = dataset.get_cifar_data()

    for idx, batch in enumerate(trainloader):
        imgs, labels = batch
        print(imgs[0], labels[0])
        break

    lr = 1e-4
    last_layer_only = False

    # Training pre-trained model
    neural_trainer.train_net(trainloader, testloader,
                             pretrained=True, net=net, num_classes=10,
                             num_epochs=200, save=False, lr=lr,
                             last_layer_only=last_layer_only)

    # Training from scratch
    neural_trainer.train_net(trainloader, testloader,
                             num_epochs=200, save=False, num_classes=10,
                             lr=lr)
    


if __name__ == "__main__":
    main()
