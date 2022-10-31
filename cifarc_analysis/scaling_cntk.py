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


BS = 500

def nt_kernel(pair1, pair2, kernel_fn_batched):

    len_1 = len(pair1)
    len_2 = len(pair2)

    if len(pair1) % 1000 != 0:
        pad = (len_1 // 1000 + 1) * 1000 - len_1
        pair1 = np.concatenate([pair1, np.zeros((pad, 32, 32, 3))], axis=0)
    if len(pair2) % 1000 != 0:
        pad = (len_2 // 1000 + 1)* 1000 - len_2
        pair2 = np.concatenate([pair2, np.zeros((pad, 32, 32, 3))], axis=0)

    out = np.array(kernel_fn_batched(pair2,
                                     pair1,
                                     fn='ntk').ntk,
                   dtype='float32')
    out = out[:len_2, :len_1]
    out = out.T
    return out#torch.from_numpy(out)


def solve_exact(X_train, X_test,
                y_train, y_test,
                kernel_fn, fn, reg=0, save=False):

    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1,
                                 batch_size=BS, store_on_device=False)

    start = time.time()
    #K_train = kernel_fn_batched(X_train, X_train, fn)
    K_train = nt_kernel(X_train, X_train, kernel_fn_batched)

    end = time.time()
    #print("KERNEL TRAIN MATRIX shape and elementwise bounds: ")
    #print(K_train.shape)
    #print(K_train.max(), K_train.min())
    #print("Time to construct K train: ", end - start)
    #K_train += np.eye(len(K_train)) * reg
    sol = solve(K_train, y_train)
    if save:
        hickle.dump(sol, 'saved_models/convnet_cifar10.h')

    del K_train
    torch.cuda.empty_cache()

    start = time.time()
    #K_test = kernel_fn_batched(X_test, X_train, fn)
    K_test = nt_kernel(X_test, X_train, kernel_fn_batched)
    end = time.time()
    #print("KERNEL TEST MATRIX shape and elementwise bounds: ")
    #print(K_test.shape)
    #print("Time to construct K test: ", end - start)

    acc = get_acc(sol, K_test, y_test)
    return sol, acc


def get_acc(sol, K, y):
    preds = K @ sol
    preds = np.argmax(preds, axis=-1)

    acc = np.mean(preds == np.argmax(y, axis=-1))
    #print("Acc Exact Solve: ", acc)
    return acc

def get_preds(sol, X_train, X, kernel_fn, fn):
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1,
                                 batch_size=BS, store_on_device=False)
    #K = kernel_fn_batched(X, X_train, fn)
    K = nt_kernel(X, X_train, kernel_fn_batched)
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
    sol = hickle.load('saved_models/convnet_cifar10.h')

    #sol, _ = solve_exact(X_train, X_test,
    #                     y_train, y_test,
    #                     kernel_fn, fn, reg=0, save=False)

    shifts = ['impulse_noise']#['fog', 'brightness']
    #, 'brightness', 'defocus_blur',
    #'elastic_transform', 'fog', 'frost',
    #'gaussian_blur', 'gaussian_noise', 'glass_blur',
    #'impulse_noise', 'jpeg_compression', 'motion_blur',
    #'pixelate', 'saturate', 'shot_noise',
    #'snow', 'spatter', 'speckle_noise', 'zoom_blur']


    for shift in sorted(shifts):
        row = [shift]
        images, one_hot = dl.load_cifar_c(num_train=50000,
                                          shift_name=shift,
                                          flatten=flatten,
                                          label_set=None)
        Xn_train = images[-10000:-1000]
        yn_train = one_hot[-10000:-1000]
        Xn_test = images[-1000:]
        yn_test = one_hot[-1000:]

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
                                              yt_subset, yn_test,
                                              kernel_fn, fn)
                seed_base.append(baseline_acc)
                f1_preds = get_preds(sol, X_train, Xt_subset, kernel_fn, fn)

                sol_c, _ = solve_exact(Xt_subset, Xn_test,
                                       yt_subset - f1_preds, yn_test,
                                       kernel_fn, fn)

                c1_preds = get_preds(sol_c, Xt_subset, Xn_test, kernel_fn, fn)
                c2_preds = get_preds(sol, X_train, Xn_test, kernel_fn, fn)
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
        path = 'scaling_law_results/cntk_' + shift + '_'
        hickle.dump(subsets, path + 'subsets.h')
        hickle.dump(baseline_results, path + 'baseline_results.h')
        hickle.dump(tf_results, path + 'tf_results.h')

        
if __name__ == "__main__":
    main()
