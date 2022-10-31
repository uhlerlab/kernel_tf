import numpy as np
import eigenpro
import torch
from neural_tangents import stax
import neural_tangents as nt
import classic_kernel
import time
import visdom
from copy import deepcopy
import gc


vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')


def ntk_kernel(pair1, pair2):
    print("Starting: ", pair1.shape[0], pair2.shape[0])
    start = time.time()
    bs = 100000
    pairs = torch.split(pair2, bs)
    outs = []
    for p in pairs:
        out = kernel(pair1, p)
        outs.append(out)
        del out
        gc.collect()
    out = torch.cat(outs, dim=1)
    end = time.time()
    print("Finished, Time:\t", end - start)
    return out


def nonlinearity(act_name='relu'):
    if act_name == 'relu':
        return stax.ABRelu(b=np.sqrt(2), a=0)
    elif act_name == 'erf':
        return stax.Erf(a = np.sqrt(np.pi/2 * 1/np.arcsin(2/3)))


def net(d, c=1, b_std=0, act_name='relu'):
    _, _, kernel_fn = stax.serial(
        stax.Dense(1, W_std=np.sqrt(d), b_std=b_std),
        nonlinearity(act_name=act_name),
        stax.Dense(1, W_std=np.sqrt(c)),
        nonlinearity(act_name=act_name),
        stax.Dense(1, W_std=np.sqrt(c)),
        nonlinearity(act_name=act_name),
        stax.Dense(1, W_std=np.sqrt(c)),
        nonlinearity(act_name=act_name),
        stax.Dense(1, W_std=np.sqrt(c)),
        nonlinearity(act_name=act_name),
        stax.Dense(1, W_std=np.sqrt(c))       
    )
    return kernel_fn

d = 3072
kernel_fn = net(d)


def k0(z):
    return 1 / np.pi * (np.pi - torch.acos(z))



def k1(z):
    return 1/ np.pi * (z * (np.pi - torch.acos(z)) \
                       + torch.sqrt(1. - torch.pow(z, 2)))
    


def kernel(pair1, pair2, depth=5):

    pair1 = pair1.cuda()
    pair2 = pair2.cuda()
    S = pair1 @ pair2.transpose(1, 0)
    ones = torch.cuda.FloatTensor(S.shape).fill_(1)
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
    K = K.cpu().float() / (depth + 1)

    del S
    gc.collect()
    torch.cuda.empty_cache()
    return K


def train(train_X, train_y, test_X, test_y, name='weights', num_epochs=30,
          compute_eval=False, laplace=False):

    SEED = 17
    np.random.seed(SEED)

    use_cuda = torch.cuda.is_available()

    if not laplace:
        device = 'cpu'
        model = eigenpro.FKR_EigenPro(ntk_kernel, train_X, train_y.shape[-1], device=device,
                                      name=name,
                                      compute_eval=compute_eval)
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        kernel_fn = lambda x,y: classic_kernel.laplacian(x, y, bandwidth=10)
        model = eigenpro.FKR_EigenPro(kernel_fn, train_X, train_y.shape[-1], device=device,
                                      name=name,
                                      compute_eval=compute_eval)

    MAX_EPOCHS = num_epochs
    epochs = list(range(MAX_EPOCHS))
    model.fit(train_X, train_y, test_X, test_y, epochs=epochs, mem_gb=24)


def get_acc(train_X, X, y, weight, laplace=False):
    if not laplace:
        device = 'cpu'
        model = eigenpro.FKR_EigenPro(ntk_kernel, train_X, y.shape[-1], device=device)
    else:
        device = 'cpu'
        kernel_fn = lambda x,y: classic_kernel.laplacian(x, y, bandwidth=10)
        model = eigenpro.FKR_EigenPro(kernel_fn, train_X, y.shape[-1], device=device)

    tensor_X = model.tensor(X)
    bs = 8000
    pairs = torch.split(tensor_X, bs)
    preds = []
    for p in pairs:
        pred = model.forward(p, weight=weight).numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    print(preds.shape)
    p_class = np.argmax(preds, axis=-1)
    y_class = np.argmax(y, axis=-1)
    return np.mean(y_class == p_class)


def get_preds(train_X, train_y, X, weight, bs=8000):
    device = 'cpu'
    model = eigenpro.FKR_EigenPro(ntk_kernel, train_X, train_y.shape[-1], device=device)

    tensor_X = model.tensor(X)

    pairs = torch.split(tensor_X, bs)
    preds = []
    for p in pairs:
        pred = model.forward(p, weight=weight).numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    return preds
