


import os
import torch
import numpy as np
from numba import jit
from numba import float64
from numba import int64
#import tqdm


hp_default_value={'model':'resnet',
                  'model_scale':'50',
                  'lr':1e-6,
                  'bs':64,
                  'epochs':20,
                  'pretrained':True,
                  'augmentation':True,
                  'is_multilabel':False,
                  'image_size':(224,224),
                  'crop':None,
                  'prevalence_setting':'separate',
                  'save_model':False,
                  'num_workers':2,
                  'num_classes':1

}


@jit((float64[:], int64), nopython=True, nogil=True)
def ewma(arr_in, window):
    r"""Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    #>>> import pandas as pd
    #>>> a = np.arange(5, dtype=float)
    #>>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    #>>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma

def expon_smoothing(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out

def get_cur_version(dir_path):
    i = 0
    while os.path.exists(dir_path+'/version_{}'.format(i)):
        i+=1
    return i


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def test_func(args,model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,args.num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()

def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()

import copy
def make_temp_L(L):
  temp_L = copy.deepcopy(L)
  temp_L["accs"].append(temp_L["accs"][-1])
  temp_L["losses"].append(temp_L["losses"][-1])
  temp_L["lrs_to_plot"].append(temp_L["lrs_to_plot"][-1])
  temp_L["num_samples"].append(temp_L["num_samples"][-1] +(temp_L["num_samples"][-1]-temp_L["num_samples"][-2]))
  return temp_L