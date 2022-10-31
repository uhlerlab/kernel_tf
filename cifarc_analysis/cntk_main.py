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


BS = 500

def solve_laplace_exact(X_train, X_test,
                        y_train, y_test,
                        reg=0, save=False):

    X_train = torch.from_numpy(np.array(X_train))
    X_test = torch.from_numpy(np.array(X_test))
    start = time.time()
    K_train = kernel.laplacian(X_train, X_train, bandwidth=10).numpy()
    end = time.time()
    print("KERNEL TRAIN MATRIX shape and elementwise bounds: ")
    print(K_train.shape)
    print(K_train.max(), K_train.min())
    print("Time to construct K train: ", end - start)
    #K_train += np.eye(len(K_train)) * reg
    sol = solve(K_train, y_train)

    del K_train
    torch.cuda.empty_cache()

    start = time.time()
    K_test = kernel.laplacian(X_test, X_train, bandwidth=10).numpy()

    end = time.time()
    print("KERNEL TEST MATRIX shape and elementwise bounds: ")
    print(K_test.shape)
    print(K_test.max(), K_test.min())
    print("Time to construct K test: ", end - start)

    acc = get_acc(sol, K_test, y_test)
    return sol, acc

def solve_exact(X_train, X_test,
                y_train, y_test,
                kernel_fn, fn, reg=0, save=False):

    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1,
                                 batch_size=BS, store_on_device=False)

    start = time.time()
    K_train = kernel_fn_batched(X_train, X_train, fn)

    end = time.time()
    print("KERNEL TRAIN MATRIX shape and elementwise bounds: ")
    print(K_train.shape)
    print(K_train.max(), K_train.min())
    print("Time to construct K train: ", end - start)
    #K_train += np.eye(len(K_train)) * reg
    sol = solve(K_train, y_train)
    if save:
        hickle.dump(sol, 'saved_models/convnet_cifar10.h')

    del K_train
    torch.cuda.empty_cache()

    start = time.time()
    K_test = kernel_fn_batched(X_test, X_train, fn)

    end = time.time()
    print("KERNEL TEST MATRIX shape and elementwise bounds: ")
    print(K_test.shape)
    print(K_test.max(), K_test.min())
    print("Time to construct K test: ", end - start)

    acc = get_acc(sol, K_test, y_test)
    return sol, acc


def get_acc(sol, K, y):
    preds = K @ sol
    preds = np.argmax(preds, axis=-1)

    acc = np.mean(preds == np.argmax(y, axis=-1))
    print("Acc Exact Solve: ", acc)
    return acc

def get_preds(sol, X_train, X, kernel_fn, fn):
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1,
                                 batch_size=BS, store_on_device=False)
    K = kernel_fn_batched(X, X_train, fn)
    return K @ sol


def main():
    SEED = 17
    np.random.seed(SEED)
    random.seed(SEED)

    flatten = False

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
    print("Train labels: ", np.sum(y_train, axis=0))
    print("Training set: ", X_train.shape, y_train.shape)
    print("Test set: ", X_test.shape, y_test.shape)

    if flatten:
        d = X_train[0].shape
    else:
        w, h, c = X_train[0].shape
        d = w * h * c

    kernel_fn = net.conv_net(1)

    fn = 'ntk'
    #sol = hickle.load('./saved_models/convnet_cifar10.h')

    sol, _ = solve_exact(X_train, X_test,
                         y_train, y_test,
                         kernel_fn, fn, reg=0, save=True)

    shifts = ['contrast', 'brightness', 'defocus_blur',
              'elastic_transform', 'fog', 'frost',
              'gaussian_blur', 'gaussian_noise', 'glass_blur',
              'impulse_noise', 'jpeg_compression', 'motion_blur',
              'pixelate', 'saturate', 'shot_noise',
              'snow', 'spatter', 'speckle_noise', 'zoom_blur']


    with open('csv_log/cntk_results.txt', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['shift_name', 'direct_acc', 'baseline_acc',
                         'translated_acc', 'projected_acc'])
        for shift in sorted(shifts):
            row = [shift]
            print("==========================================================")
            images, one_hot = dl.load_cifar_c(num_train=50000,
                                              shift_name=shift,
                                              flatten=flatten,
                                              label_set=None)
            Xn_train = images[-10000:-1000]
            yn_train = one_hot[-10000:-1000]
            Xn_test = images[-1000:]
            yn_test = one_hot[-1000:]

            print("Noisy Train labels: ", np.sum(yn_train, axis=0))
            print(Xn_train.shape, yn_train.shape, Xn_test.shape, yn_test.shape)

            #"""
            kernel_fn_batched = nt.batch(kernel_fn, device_count=-1,
                                         batch_size=BS, store_on_device=False)

            print("First model predictions on Second dataset:")
            K = kernel_fn_batched(Xn_test, X_train, fn)
            direct_pred_acc = get_acc(sol, K, yn_test)
            row.append(direct_pred_acc)

            print("==========================================================")
            print("Training Directly on Second dataset:")
            sol_n, baseline_acc = solve_exact(Xn_train, Xn_test,
                                              yn_train, yn_test,
                                              kernel_fn, fn)
            row.append(baseline_acc)
            print("==========================================================")
            f1_preds = get_preds(sol, X_train, Xn_train, kernel_fn, fn)

            sol_b, _ = solve_exact(Xn_train, Xn_test,
                                   yn_train - f1_preds, yn_test,
                                   kernel_fn, fn)

            c1_preds = get_preds(sol_b, Xn_train, Xn_test, kernel_fn, fn)
            c2_preds = get_preds(sol, X_train, Xn_test, kernel_fn, fn)
            c_preds = c1_preds + c2_preds

            c_preds = np.argmax(c_preds, axis=-1)
            translated_acc = np.mean(c_preds == np.argmax(yn_test, axis=-1))
            print("Translated Acc: ", translated_acc)
            row.append(translated_acc)
            print("==========================================================")

            #"""

            f1_test_preds = get_preds(sol, X_train, Xn_test, kernel_fn, fn)
            #_, d_temp = f1_test_preds.shape
            #kernel_fn_fc = net.simple_fc(d_temp)

            _, projected_acc = solve_laplace_exact(f1_preds, f1_test_preds,
                                                   yn_train, yn_test, reg=0)
            row.append(projected_acc)
            writer.writerow(row)
            f.flush()
            #"""

if __name__ == "__main__":
    main()
