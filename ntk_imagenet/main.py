import dataset as d
import trainer as t
import torch
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

    train_data, test_data = d.get_imagenet_data()
    X_train, y_train = train_data
    X_test, y_test = test_data

    print(len(X_train), len(y_train))

    t.train(X_train, y_train, X_test, y_test)

    
if __name__ == "__main__":
    main()
