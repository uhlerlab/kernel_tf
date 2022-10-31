import numpy as np
import random
from numpy.linalg import solve, norm, inv
import time
import neural_tangents as nt
import dataloader as dl
import net
import torch
import eigenpro
import kernel
import hickle
import csv
import gc
from copy import deepcopy

BS = 500

def k0(z):
    return 1 / np.pi * (np.pi - torch.acos(z))


def k1(z):
    return 1/ np.pi * (z * (np.pi - torch.acos(z)) \
                                   + torch.sqrt(1. - torch.pow(z, 2)))


def ntk_kernel(pair1, pair2, depth=5):
    pair1 = pair1.float()
    pair2 = pair2.float()
    S = pair1 @ pair2.transpose(1, 0)
    ones = torch.FloatTensor(S.shape).fill_(1)
    S = torch.where(torch.isclose(S, ones), ones, S)
    del ones
    torch.cuda.empty_cache()
    S = torch.clamp(S, -1, 1)
    K = deepcopy(S)

    for i in range(depth):
        S = torch.clamp(S, -1, 1)
        S_ = k0(S)
        K_ = K * k1(S) + S_
        K = K_
        S = S_
        del K_, S_
        torch.cuda.empty_cache()
    K = K.float() / (depth + 1)
    del S
    gc.collect()
    torch.cuda.empty_cache()
    return K.cpu()


def solve_exact(X_train, X_test,
                y_train, y_test,
                reg=0, save=False):

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    start = time.time()
    K_train = ntk_kernel(X_train, X_train).numpy()
    end = time.time()
    #print("KERNEL TRAIN MATRIX shape and elementwise bounds: ")
    #print(K_train.shape)
    #print(K_train.max(), K_train.min())
    #print("Time to construct K train: ", end - start)
    #K_train += np.eye(len(K_train)) * reg
    sol = solve(K_train, y_train)
    if save:
        hickle.dump(sol, 'saved_models/ntk_cifar10.h')

    del K_train
    torch.cuda.empty_cache()

    start = time.time()
    K_test = ntk_kernel(X_test, X_train).numpy()

    end = time.time()
    #print("KERNEL TEST MATRIX shape and elementwise bounds: ")
    #print(K_test.shape)
    #print(K_test.max(), K_test.min())
    #print("Time to construct K test: ", end - start)

    acc = get_acc(sol, K_test, y_test)
    return sol, acc


def get_acc(sol, K, y):
    preds = K @ sol
    preds = np.argmax(preds, axis=-1)

    acc = np.mean(preds == np.argmax(y, axis=-1))
    print("Acc Exact Solve: ", acc)
    return acc

def get_preds(sol, X_train, X):
    X_train = torch.from_numpy(X_train)
    X = torch.from_numpy(X)
    K = ntk_kernel(X, X_train).numpy()
    return K @ sol


def main():
    SEED = 17
    np.random.seed(SEED)
    random.seed(SEED)

    flatten = True

    X_train, y_train, X_test, y_test = dl.load_cifar(num_train=50000,
                                                     num_test=10000,
                                                     flatten=flatten,
                                                     label_set=None)

    num_train = 50000
    num_test = 10000
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    X_train = X_train / norm(X_train, axis=1).reshape(-1, 1)
    X_test = X_test / norm(X_test, axis=1).reshape(-1,1)
    print("Train labels: ", np.sum(y_train, axis=0))
    print("Training set: ", X_train.shape, y_train.shape)
    print("Test set: ", X_test.shape, y_test.shape)


    sol = hickle.load('./saved_models/ntk_cifar10.h')

    #sol, _ = solve_exact(X_train, X_test,
    #                     y_train, y_test,
    #                     reg=0, save=True)


    shifts = ['impulse_noise']# 'contrast', 'defocus_blur'],
              #'elastic_transform', 'fog', 'frost',
              #'gaussian_blur', 'gaussian_noise', 'glass_blur',
              #'impulse_noise', 'jpeg_compression', 'motion_blur',
              #'pixelate', 'saturate', 'shot_noise',
              #'snow', 'spatter', 'speckle_noise', 'zoom_blur']



    for shift in shifts:
        row = [shift]
        images, one_hot = dl.load_cifar_c(num_train=50000,
                                          shift_name=shift,
                                          flatten=flatten,
                                          label_set=None)
        Xn_train = images[-10000:-1000]
        yn_train = one_hot[-10000:-1000]
        Xn_test = images[-1000:]
        yn_test = one_hot[-1000:]
        Xn_train = Xn_train / norm(Xn_train, axis=1).reshape(-1, 1)
        Xn_test = Xn_test / norm(Xn_test, axis=1).reshape(-1, 1)

        subsets = list(np.arange(0,len(Xn_train), len(Xn_train)//50))[1:]

        num_subsamples = 3

        baseline_results = []
        tf_results = []
        for subset in subsets:
            seed_base = []
            seed_tf = []
            for seed in range(num_subsamples):
                print("SUBSET: ", subset, "SEED: ", seed)
                start = time.time()
                indices = random.sample(list(np.arange(len(Xn_train))), subset)
                Xt_subset = Xn_train[indices]
                yt_subset = yn_train[indices]
                _, baseline_acc = solve_exact(Xt_subset, Xn_test,
                                              yt_subset, yn_test)
                seed_base.append(baseline_acc)
                f1_preds = get_preds(sol, X_train, Xt_subset)
                sol_c, _ = solve_exact(Xt_subset, Xn_test,
                                       yt_subset - f1_preds, yn_test)
                c1_preds = get_preds(sol_c, Xt_subset, Xn_test)
                c2_preds = get_preds(sol, X_train, Xn_test)
                c_preds = c1_preds + c2_preds

                c_preds = np.argmax(c_preds, axis=-1)
                translated_acc = np.mean(c_preds == np.argmax(yn_test, axis=-1))
                seed_tf.append(translated_acc)
                end = time.time()
                print("Finished: ", end - start, baseline_acc, translated_acc)
            tf_results.append(seed_tf)
            baseline_results.append(seed_base)
        print(baseline_results)
        print(tf_results)
        path = 'scaling_law_results/ntk_' + shift + '_'
        hickle.dump(subsets, path + 'subsets.h')
        hickle.dump(baseline_results, path + 'baseline_results.h')
        hickle.dump(tf_results, path + 'tf_results.h')

if __name__ == "__main__":
    main()
