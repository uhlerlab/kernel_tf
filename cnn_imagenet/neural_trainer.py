import neural_model as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from copy import deepcopy


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight,a=0,
                                       mode='fan_in',
                                       nonlinearity='relu')

def train_net(train_loader, test_loader,
              pretrained=False,
              net=None, num_classes=None,
              save=False, shift_label=False,
              num_epochs=None, lr=1e-2,
              last_layer_only=False):

    if not pretrained:
        if num_classes is not None:
            net = nn.Net(num_classes=num_classes)
        else:
            net = nn.Net()
        net.apply(init_weights)
    else:
        net = nn.PretrainedNet(net, num_classes)

    if last_layer_only:
        optimizer = optim.Adam([list(net.parameters())[-1]])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                               lr=lr)

    if num_epochs is None: 
        num_epochs = 100

    net.cuda()

    best_acc = 0.
    
    for i in range(num_epochs):
        train_loss = train_step(net, train_loader, optimizer,
                                shift_label=shift_label)
        train_acc = get_acc(net, train_loader,
                            shift_label=shift_label)
        test_acc = get_acc(net, test_loader,
                           shift_label=shift_label)
        print("Epoch: ", i+1,
              "Train Loss: ", train_loss,
              "Train Acc: ", train_acc,
              "Test Acc: ", test_acc)
        
        if test_acc > best_acc and save:
            best_acc = test_acc
            d = {}
            d['state_dict'] = net.state_dict()
            torch.save(d, 'saved_models/trained_model.pth')
            

        
def train_step(net, train_loader, optimizer, iteration=None,
               shift_label=False):
    net.train()
    train_loss = 0.

    loss_fn = torch.nn.CrossEntropyLoss()
    
    for idx, batch in enumerate(train_loader):
        #print(idx, len(train_loader))
        optimizer.zero_grad()
        inputs, labels = batch
        outputs = net(inputs.cuda())
        if shift_label:
            labels -= 1
        loss = loss_fn(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy().item() * len(batch)

    return train_loss / len(train_loader.dataset)

 
def get_acc(net, loader,
            shift_label=False):
    net.eval()
    acc = 0
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        if shift_label:
            labels -= 1
        outputs = net(inputs.cuda()).cpu().data.numpy()
        preds = np.argmax(outputs, axis=-1)
        acc += np.sum(preds == labels.numpy())
    return acc / len(loader.dataset)
