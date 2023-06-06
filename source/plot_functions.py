import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import cm

def plot_confusion(y_true, y_pred):
    '''plots the confusion matrix for multilabel classification 
    '''
    a = unique_labels(y_true, y_pred)
    plt.figure(figsize=(14,10))
    cm = confusion_matrix(y_true, y_pred, labels=a)
    cm_f = cm / cm.sum(axis=1, keepdims=True)
    annot = np.empty_like(cm).astype(str)
    nr, nc = cm.shape
    for i in range(nr):
        for j in range(nc):
            c = cm[i,j]; f = cm_f[i,j]
            annot[i,j] = "%.2f\n%d" % (f,c)
    cm = pd.DataFrame(cm, index=a, columns=a)
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    ax.set_ylim(0,len(a))
    ax.invert_yaxis()
    ax.set_xlabel('predicted mineral')
    ax.set_ylabel('true mineral')
    plt.show()

def plot_predicted_mineralogy(hypr_rgb, pred_mineral, LE, figsize=(16,8), cmap='cividis'):
    '''plots the rgb image of the hyperspectral image and the predicted mineralogy
    side by side.
    '''
    formatter = plt.FuncFormatter(lambda val, loc: LE.inverse_transform(np.array([val]))[0])
    k = len(LE.classes_)
    v = cm.get_cmap(cmap, k)

    _, axes = plt.subplots(ncols=2, figsize=figsize)
    axes[0].imshow(hypr_rgb)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im = axes[1].imshow(pred_mineral, cmap=v, vmin=-0.5, vmax=k-0.5)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.colorbar(im, ticks=np.arange(0,k), format=formatter, ax=axes.ravel().tolist(), pad=0.02, shrink=0.5)
    plt.show()