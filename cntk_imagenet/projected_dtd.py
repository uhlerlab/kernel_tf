import dataset as d
import trainer as t
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pickle
from numpy.linalg import norm


def main():
    SEED = 17
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Set path to Imagenet32 data
    imagenet_path = None
    train_data, _ = d.get_imagenet_data(path=imagenet_path)
    X_train, y_train = train_data
    
    X_train = X_train[:1280000]
    y_train = y_train[:1280000]

    weight = pickle.load(open('saved_models/cntk_6_epochs_full_imagenet.p', 'rb'))

    # Set path to DTD data
    dtd_path = None
    
    train_loader, test_loader = d.get_dtd_data(path=dtd_path)
    Xt_train, yt_train = d.normalized_numpy_data(train_loader, 47,
                                                 shift=False)
    Xt_test, yt_test = d.normalized_numpy_data(test_loader, 47,
                                               shift=False)
    pred_train = t.get_preds(X_train, y_train, Xt_train, weight, bs=4000)
    pred_test = t.get_preds(X_train, y_train, Xt_test, weight, bs=4000)

    
    Xt_train = Xt_train.astype('float32')
    Xt_test = Xt_test.astype('float32')

    yt_train = yt_train.astype('float32')
    yt_test = yt_test.astype('float32')

    pred_train = pred_train.astype('float32')
    pred_test = pred_test.astype('float32')
    
    print("Transformed Features: ", pred_train.shape, pred_test.shape)

    t.train(pred_train, yt_train, pred_test, yt_test, name='dtd_proj',
            num_epochs=100, compute_eval=True, laplace=True)

    t.train(Xt_train, yt_train, Xt_test, yt_test, name='dtd_baseline',
            num_epochs=30, compute_eval=True)#, laplace=True)


if __name__ == "__main__":
    main()
