import dataset as d
import trainer as t
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pickle
from numpy.linalg import norm, solve
import classic_kernel


def main():
    SEED = 17
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # Set path to imagenet data
    imagenet_path = None
    train_data, _ = d.get_imagenet_data(path=imagenet_path)
    
    X_train, y_train = train_data

    # Model was trained on only 1.28 mill. ex exactly 
    X_train = X_train[:1280000]
    y_train = y_train[:1280000]

    # Using model trained for 6 epochs on Imagenet32
    weight = pickle.load(open('saved_models/cntk_6_epochs_full_imagenet.p', 'rb'))    

    # Set path to cifar data
    cifarpath = None
    train_loader, test_loader = d.get_cifar_data(path=cifarpath)
    Xt_train, yt_train = d.normalized_numpy_data(train_loader, 10)
    Xt_test, yt_test = d.normalized_numpy_data(test_loader, 10)
    
    pred_train = t.get_preds(X_train, y_train, Xt_train, weight, bs=4000)
    pred_test = t.get_preds(X_train, y_train, Xt_test, weight, bs=4000)

    Xt_train = Xt_train.astype('float32')
    Xt_test = Xt_test.astype('float32')

    pred_train = pred_train.astype('float32')
    pred_test = pred_test.astype('float32')

    yt_train = yt_train.astype('float32')
    yt_test = yt_test.astype('float32')

    print("Source Prediction Shapes: ", pred_train.shape, pred_test.shape)

    pred_train = torch.from_numpy(pred_train)
    pred_test = torch.from_numpy(pred_test)
    K_train = classic_kernel.laplacian(pred_train, pred_train, bandwidth=10)
    K_train = K_train.numpy()
    sol = solve(K_train, yt_train).T
    K_test = classic_kernel.laplacian(pred_train, pred_test, bandwidth=10)
    K_test = K_test.numpy()
    preds = np.argmax((sol @ K_test).T, axis=-1)
    labels = np.argmax(yt_test, axis=-1)
    acc = np.mean(preds == labels)
    print("Projected Acc: ", acc)


if __name__ == "__main__":
    main()
