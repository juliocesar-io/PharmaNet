from __future__ import division, print_function

import argparse
import copy
import os
import random
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.pyplot import imshow

from utils.arguments import get_args

args = vars(get_args())


def plot_loss(d_losses, g_losses, save_dir, num_epoch=args["epochs"], save=True, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0,args["epochs"])
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Train')
    plt.plot(g_losses, label='Test')
    plt.legend()
    
    # Save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'losses_{:d}'.format(num_epoch) + '.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()

def plot_nap(d_losses, g_losses, f_losses, td_losses, tg_losses, tf_losses, save_dir, num_epoch=args["epochs"], save=True, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0,args["epochs"])
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses), np.max(f_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch))
    plt.ylabel('metric values')
    plt.plot(d_losses, label='Test nap')
    plt.plot(g_losses, label='Test AUC')
    plt.plot(f_losses, label='Test accuracy')
    plt.plot(td_losses, label='Train nap')
    plt.plot(tg_losses, label='Train AUC')
    plt.plot(tf_losses, label='Train accuracy')
    plt.legend()
    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'nap{:d}'.format(num_epoch) + '.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()
