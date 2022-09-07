import numpy as np
import eigenpro
import torch
from neural_tangents import stax
import neural_tangents as nt
import classic_kernel
import time



def nonlinearity(act_name='relu'):
    if act_name == 'relu':
        return stax.ABRelu(b=np.sqrt(2), a=0)
    elif act_name == 'erf':
        return stax.Erf(a = np.sqrt(np.pi/2 * 1/np.arcsin(2/3)))


def net(d, c=1, b_std=0, act_name='relu'):
    _, _, kernel_fn = stax.serial(
        stax.Dense(1, W_std=np.sqrt(d), b_std=b_std),
        nonlinearity(act_name=act_name),
        stax.Dense(1, W_std=np.sqrt(c))
    )
    return kernel_fn

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


kernel_fn = conv_net()
kernel_fn_batched = nt.batch(kernel_fn, device_count=-1,
                             batch_size=500, store_on_device=False)

def ntk_kernel(pair1, pair2):
    print("Starting\t Shape: ", pair1.shape[0], pair2.shape[0])
    start = time.time()
    bs = 10000
    pairs = torch.split(pair2, bs)
    outs = []
    for p in pairs:
        out = nt_kernel(pair1, p)
        outs.append(out)
    out = torch.cat(outs, dim=1)
    end = time.time()
    print("Finished\t Time: ", end - start, "\tMax Val:\t", out.max())
    return out


def nt_kernel(pair1, pair2):

    pair1 = pair1.numpy().reshape(-1, 3, 32, 32)
    pair2 = pair2.numpy().reshape(-1, 3, 32, 32)

    pair1 = np.rollaxis(pair1, 1, 4)
    pair2 = np.rollaxis(pair2, 1, 4)

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

    # EigenPro requires K(x,x) = 1
    # This is an approximate normalization based on a data subset.
    out /= 52 
    torch.cuda.empty_cache()
    return torch.from_numpy(out)


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
    
    # The batch size below may need to be tuned depending on GPU memory.
    # This should work for GPUs with 12GB memory.
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
