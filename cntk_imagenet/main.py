import dataset as d
import trainer as t
import torch
import random
import numpy as np


def main():
    SEED = 17
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Path to ImageNet32 data
    path = None
    train_data, test_data = d.get_imagenet_data(path=path)
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Model in paper was trained on 1.28 mill ex exactly.
    X_train = X_train[:1280000]
    y_train = y_train[:1280000]

    print(len(X_train), len(y_train))

    t.train(X_train, y_train, X_test, y_test)

    
if __name__ == "__main__":
    main()
