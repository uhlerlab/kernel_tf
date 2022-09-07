import dataset as d
import trainer as t
import torch
import torchvision
import torchvision.transforms as transforms
from neural_tangents import stax
import neural_tangents as nt
import random
import numpy as np
import pickle
from numpy.linalg import norm, solve
import classic_kernel



def nonlinearity(act_name='relu'):
    if act_name == 'relu':
        return stax.ABRelu(b=np.sqrt(2), a=0)
    elif act_name == 'erf':
        return stax.Erf(a = np.sqrt(np.pi/2 * 1/np.arcsin(2/3)))


def conv_net(c=1., b_std=0., act_name='relu'):
    pad = "SAME"
    _, _, kernel_fn = stax.serial(
        stax.Conv(1, (3,3), strides=(2,2), W_std=np.sqrt(9), padding=pad),
        nonlinearity(),
        stax.Conv(1, (3,3), strides=(2,2), W_std=np.sqrt(9), padding=pad),
        nonlinearity(),
        stax.Conv(1, (3,3), strides=(2,2), W_std=np.sqrt(9), padding=pad),
        nonlinearity(),
        stax.Conv(1, (3,3), strides=(2,2), W_std=np.sqrt(9), padding=pad),
        nonlinearity(),
        stax.Conv(1, (3,3), strides=(2,2), W_std=np.sqrt(9), padding=pad),
        nonlinearity(),
        stax.Conv(1, (3,3), strides=(1,1), W_std=np.sqrt(9), padding=pad),
        stax.Flatten(),
        stax.Dense(1, W_std=c)
    )
    return kernel_fn


def laplace_solve(X_train, y_train, X_test, y_test):
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    K_train = classic_kernel.laplacian(X_train, X_train, bandwidth=10)
    K_train = K_train.numpy()
    sol = solve(K_train, y_train).T
    K_test = classic_kernel.laplacian(X_train, X_test, bandwidth=10)
    K_test = K_test.numpy()
    preds = np.argmax((sol @ K_test).T, axis=-1)
    labels = np.argmax(y_test, axis=-1)
    acc = np.mean(preds == labels)
    return acc


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
    return out

def main():
    SEED = 17
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Set ImageNet32 path
    imagenet_path = None
    train_data, _ = d.get_imagenet_data()
    X_train, y_train = train_data
    X_train = X_train[:1280000]
    y_train = y_train[:1280000]

    weight = pickle.load(open('saved_models/cntk_6_epochs_full_imagenet.p', 'rb'))    

    # Set CIFAR10 path
    cifar_path = None
    train_loader, test_loader = d.get_cifar_data(path=cifar_path)
    
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

    Xt_train = Xt_train.reshape(-1, 3, 32, 32)
    Xt_test = Xt_test.reshape(-1, 3, 32, 32)

    Xt_train = np.rollaxis(Xt_train, 1, 4)
    Xt_test = np.rollaxis(Xt_test, 1, 4)
    
    print("Transformed Features: ", pred_train.shape, pred_test.shape)

    tf_results = []
    baseline_results = []
    
    subsets = list(range(0, 50000, 1000))[1:]
    num_random_samples = 3

    kernel_fn = conv_net()
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1,
                                 batch_size=500, store_on_device=False)
    
    for s in subsets:
        seed_tf_results = []
        seed_base_results = []
        for i in range(num_random_samples):
            print("SEED: ", i, "SUBSET: ", s)
            indices = random.sample(list(np.arange(len(Xt_train))), s)
            Xt_subset = Xt_train[indices]
            yt_subset = yt_train[indices]

            pred_subset = pred_train[indices]

            acc = laplace_solve(pred_subset, yt_subset, pred_test, yt_test)
            seed_tf_results.append(acc)
            
            bs = 5000
            pairs = torch.split(torch.from_numpy(Xt_subset), bs)
            outs = []
            for p in pairs:
                out = nt_kernel(p.numpy(), Xt_subset, kernel_fn_batched)
                outs.append(out)
            K_train = np.concatenate(outs, axis=0)
            sol = solve(K_train, yt_subset).T
            bs = 5000
            pairs = torch.split(torch.from_numpy(Xt_subset), bs)
            outs = []
            for p in pairs:
                out = nt_kernel(p.numpy(), Xt_test, kernel_fn_batched)
                outs.append(out)
            K_test = np.concatenate(outs, axis=0)
            preds = np.argmax((sol @ K_test).T, axis=-1)
            labels = np.argmax(yt_test, axis=-1)
            acc = np.mean(preds == np.argmax(yt_test, axis=-1))
            seed_base_results.append(acc)
            
        tf_results.append(seed_tf_results)
        baseline_results.append(seed_base_results)
    print(tf_results)
    print(baseline_results)


if __name__ == "__main__":
    main()
