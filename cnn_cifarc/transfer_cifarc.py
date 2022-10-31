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
import dataloader as dl
from torch.utils.data import DataLoader
from copy import deepcopy
import csv


def main():
    SEED = 17
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    net = nm.Net(num_classes=10)
    fname = 'saved_models/trained_model.pth'
    d = torch.load(fname)
        
    net.load_state_dict(d['state_dict'])

    shifts = ['contrast', 'brightness', 'defocus_blur',
              'elastic_transform', 'fog', 'frost',
              'gaussian_blur', 'gaussian_noise', 'glass_blur',
              'impulse_noise', 'jpeg_compression', 'motion_blur',
              'pixelate', 'saturate', 'shot_noise',
              'snow', 'spatter', 'speckle_noise', 'zoom_blur']

    with open('csv_logs/cnn_results.txt', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['shift_name',
                         'source_acc',
                         'baseline_acc',
                         'tf_acc'])
        for shift in sorted(shifts):
            row = [shift]
            images, one_hot = dl.load_cifar_c(num_train=50000,
                                              shift_name=shift,
                                              flatten=False)
    
            Xn_train = images[-10000:-1000].astype('float32')
            yn_train = one_hot[-10000:-1000]

            
            yn_train = np.argmax(yn_train, axis=-1)
            Xn_test = images[-1000:].astype('float32')
            yn_test = one_hot[-1000:]
            yn_test = np.argmax(yn_test, axis=-1)
            
            print(Xn_train.shape, yn_train.shape, Xn_test.shape, yn_test.shape)
            trainset = dataset.ImagenetDataset(Xn_train, yn_train)
            testset = dataset.ImagenetDataset(Xn_test, yn_test)
            trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
            testloader = DataLoader(testset, batch_size=1024, shuffle=False)
            
            source_acc = neural_trainer.get_acc(net.cuda(), testloader)
            row.append(source_acc)
            
            #base_acc = neural_trainer.train_net(trainloader, testloader,
            #                                    num_epochs=200, save=False, num_classes=10,
            #                                    lr=1e-4)
            #print("Base Acc: ", base_acc)            
            #row.append(base_acc)
            row.append(0)
            
            print("Source Acc: ", source_acc)
            tf_acc = neural_trainer.train_net(trainloader, testloader,
                                              pretrained=True, net=deepcopy(net), num_classes=10,
                                              num_epochs=200, save=False, lr=1e-4)
            print("Tf Acc: ", tf_acc)
            row.append(tf_acc)

            writer.writerow(row)
            f.flush()
            
    
if __name__ == "__main__":
    main()
