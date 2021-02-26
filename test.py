rom __future__ import division, print_function


import copy
import csv
import datetime
import os
import pdb
import random
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_loader import SMILESDataset
from utils.arguments import get_args
from utils.plots import plot_loss, plot_nap
from utils.metrics import pltmap, pltauc, norm_ap
from models.Model import Model
from tqdm import tqdm

print("PyTorch Version: ",torch.__version__)

################################################################

## Parameters

args = vars(get_args())
torch.manual_seed(args["seed"])
np.random.seed(args["seed"])

#################################################################

## GPU Available
ngpu = args["ngpu"]
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(ngpu)
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################################################################

## Data for test
test_file = args["test_file"]
data_test = pd.read_csv(test_file)
print('Data for test:', len(data_test['Smiles']),'\n') 
experiment_path = os.path.join(args["results_dir"],
                                args["results_path"])
dataset = {}
for folder in range(1,5):
    # Creating Saving Folder
    saving_path = os.path.join(experiment_path, 'model{}'.format(folder))

    ## Loading the model

    model_path = os.path.join(saving_path, 'model.pth')
    trained_model = torch.load(model_path) 

    ## Embedding Functions
    char_to_int = trained_model['charset']

    vocab_size = len(char_to_int)
    embed = trained_model['embed']
    

    # ------------------- DATASET ------------------------

    dataset[folder] = SMILESDataset(data_test, vocab_size, char_to_int, embed,
                                    args["neighbours"], args["padding"])

## Class details
num_classes = max(data_test.Label)+1
targets_class = []

for i in range(num_classes):
    pos = np.where(data_test.Label == i)
    targets_class.append(data_test.Label[list(pos)[0][0]])

#------------------ Model Initialization ------------------------

hidden_size  = args["hidden_size"]
model = Model(vocab_size,
            num_classes,
            hidden_size,
            args["bidireccional"],
            args["num_layers"],
            args["kernel_size"]).to(device)

def get_trained_model(net, num_model):
    saving_path = os.path.join(experiment_path, 'model{}'.format(num_model))
    model_path = os.path.join(saving_path, 'model.pth')
    trained_model = torch.load(model_path) 
    return trained_model['model']

# ---------------- TRAIN AND TEST DATA PREPARATION ---------------

# Create training and validation dataloaders
balanced_batch = args["balanced_loader"]

def make_weights_for_balanced_classes(data, nclasses):                        
    count = [0] * nclasses                                                      
    for item in data:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(data)                                              
    for idx, val in enumerate(data):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight  

# Create training and validation datasets

def make_test_loader(test_datasets, num_classes):
    print("Loading test samples...")
    if balanced_batch:
        # For unbalanced test dataset we create a weighted sampler                     
        weights_test = make_weights_for_balanced_classes(list(data_test.Smiles),
                                                              num_classes)
        weights_test = torch.DoubleTensor(weights_test)
        sampler_test = torch.utils.data.sampler.WeightedRandomSampler(weights_test,
                                                                      len(weights_test))

        test_loader = DataLoader(test_datasets, 
                                batch_size= args["batch_size"],
                                sampler = sampler_test, 
                                num_workers= args["workers"])
    else:

        test_loader = DataLoader(test_datasets, 
                                batch_size= args["batch_size"],
                                drop_last=False ,
                                shuffle= False,
                                num_workers= args["workers"])
    return test_loader

# -------------------- TEST FUNCTIONS -------------------

def test(test_loader, targets_class, net, num_classes, saving_path):
    
    net.eval()  

    # Iterate over data
    batch = {}

    for num_model in range(1,5):
        trained_model = get_trained_model(net, num_model)
        net.load_state_dict(trained_model)
        first = True
        loader = test_loader[num_model]
        lab =  np.array([])
        smiles =  np.array([])

        for inputs, smile, labels in tqdm(loader):
            inputs  = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # track history if only in train
            with torch.set_grad_enabled(False):

                # Get model outputs and calculate loss
                outputs= net(inputs)
                if first:
                    batch[num_model] = outputs
                    first = False
                else:
                    batch[num_model] = torch.cat((batch[num_model],
                                                    outputs),
                                                    axis = 0)
                labels_array = labels.clone().cpu()
                labels_array = labels_array.detach().numpy()
                lab = np.append(lab, labels_array)
                smiles = np.append(smiles, smile)
    
    # Output modification for mAP computing
    
    outputs = torch.mean(torch.stack(list(batch.values())), axis = 0)
    outputs_array = outputs.clone().cpu()
    outputs_array = F.softmax(outputs_array, dim = 1)
    maps = outputs_array.detach().numpy()

    # Model predictions 
    _, preds = torch.max(outputs, 1)

    # Statistics

    # Accuracy
    running_corrects = torch.sum(preds.cpu() == torch.Tensor(lab))

    # Accuracy
    epoch_acc = running_corrects.double() / len(test_loader[1].dataset)

    # NAP
    epoch_nap, epoch_f = norm_ap(maps,lab)                
    
    # AUC
    epoch_ap, _ = pltmap(maps, lab, num_classes)

    # AUC
    epoch_auc = pltauc(maps, lab, num_classes)
    
    predictions = [smiles, maps, lab]
    
    return epoch_acc, epoch_ap["micro"], epoch_nap[-1], epoch_f[-1], epoch_auc["micro"], predictions

# -------------------- TRAIN AND TEST FUNCTIONS -------------------
     


def main():
    
    since = time.time()
    print('Testing PharmaNet')
    loader = {}
    for folder in range(1,5):
        loader[folder] = make_test_loader(dataset[folder], num_classes)
    
        
    acc, aps, naps, fmeasure, auc, predictions_map = test(loader,
                                                     targets_class, model,
                                                     num_classes, saving_path)

    time_elapsed = time.time() - since
    print('Test completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                     time_elapsed % 60))
    print('Test NAP: {:4f}'.format(naps))
    print('Test NAP: {:4f}'.format(aps))
    print('Test ACC: {:4f}'.format(acc))
    print('Test AUC: {:4f}'.format(auc))
    print('Test F-measure: {:4f}\n'.format(fmeasure))
    
    torch.save({
                'prediction': predictions_map
                }, os.path.join(saving_path, 'test_predictions.pth'))

if __name__ == '__main__':

    main()
