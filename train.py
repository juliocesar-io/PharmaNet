from __future__ import division, print_function

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
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models
from data.data_loader import SMILESDataset
from utils.arguments import get_args
from utils.plots import plot_loss, plot_nap
from utils.metrics import pltmap, pltauc, norm_ap_optimized
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

if device == 'cuda':
    torch.cuda.manual_seed(args["seed"])


# Creating Saving Folder
saving_path = os.path.join(args["results_dir"],
                           args["results_path"])
if not os.path.exists(saving_path):
    os.makedirs(saving_path,0o777)

#################################################################

#DATA
train_path = args["train_path"]
A = pd.read_csv(train_path+'Smiles_1.csv')
B = pd.read_csv(train_path+'Smiles_2.csv')
C = pd.read_csv(train_path+'Smiles_3.csv')
D = pd.read_csv(train_path+'Smiles_4.csv')

cross_val = args["cross_val"]
if cross_val == 1:
    data_train = pd.concat([A,B,C], ignore_index = True)
    data_test = D
elif cross_val == 2:
    data_train = pd.concat([A,C,D], ignore_index = True)
    data_test = B
elif cross_val == 3:
    data_train = pd.concat([A,B,D], ignore_index = True)
    data_test = C 
else:
    data_train = pd.concat([B,C,D], ignore_index = True)
    data_test = A    

print('Data for train:', len(data_train['Smiles'])) 
print('Data for test:', len(data_test['Smiles'])) 

################################################################

## Embedding Functions

charset = set( "".join(list(data_train.Smiles)) + "".join(list(data_test.Smiles)) )
vocab_size = len(charset)
char_to_int = dict((c,i) for i,c in enumerate(charset))
embed_tr = max([len(smile) for smile in data_train.Smiles])
embed_te = max([len(smile) for smile in data_test.Smiles]) 
embed = max(embed_tr,embed_te)+ args["add_val"]
print('Set of Charaters:',str(charset))
print('Num of Charsets:', vocab_size)
print('Max. length of smile:', embed)

## Class details
num_classes = max(data_train.Label)+1
targets_class = []

for i in range(num_classes):
    pos = np.where(data_test.Label == i)
    targets_class.append(data_test.Label[list(pos)[0][0]])


################################################################

#------------------ Model Initialization ------------------------

hidden_size  = args["hidden_size"]
kernel_size = args["kernel_size"]

net = Model(vocab_size,
            num_classes,
            hidden_size,
            args["bidireccional"],
            args["num_layers"],
            kernel_size
            ).to(device)

print(net)

#################################################################

# ------------------- OPTIMIZER AND LOSS ------------------------

optimizer = optim.Adam(net.parameters(), lr= args["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor = 0.1,
                                                 patience = 7)

# Setup the loss function
criterion = nn.BCELoss()


# ---------------- TRAIN AND TEST DATA PREPARATION ---------------

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets

train_datasets = SMILESDataset(data_train, vocab_size,
                               char_to_int, embed, args["neighbours"],
                               args["padding"])

test_datasets = SMILESDataset(data_test, vocab_size,
                              char_to_int, embed, args["neighbours"],
                              args["padding"])

################################################################

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

if balanced_batch:
    # For unbalanced train dataset we create a weighted sampler                     
    weights_train = make_weights_for_balanced_classes(list(data_train.Smiles),
                                                           num_classes)
    weights_train = torch.DoubleTensor(weights_train)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train,
                                                                   len(weights_train))

    train_loader = DataLoader(train_datasets, 
                            batch_size= args["batch_size"],
                            sampler = sampler_train, 
                            num_workers= args["workers"])

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

    train_loader = DataLoader(train_datasets, 
                            batch_size= args["batch_size"],
                            drop_last=True,
                            shuffle= True,
                            num_workers= args["workers"])

    test_loader = DataLoader(test_datasets, 
                            batch_size= args["batch_size"],
                            drop_last=True,
                            shuffle= True,
                            num_workers= args["workers"])

# -------------------- TRAIN AND TEST FUNCTIONS -------------------


def train(epoch):
    net.train()

    # Epoch Scores and loss
    running_loss = 0.0
    running_corrects = 0

    lab =  np.array([])
    maps = np.empty((0, num_classes))

    # Iterate over data
    for inputs, _, labels in tqdm(train_loader):
    
        inputs  = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward

        # track history if only in train
        with torch.set_grad_enabled(True):

            # Get model outputs and calculate loss
            outputs= net(inputs)
            loss = 0
            for i in range(num_classes):
                labe = labels.clone()
                labe[labels == i] = 1
                labe[labels != i] = 0
                loss += criterion(F.sigmoid(outputs[:,i].float()),
                                            labe.float().to(device))
           
            # Output modification for mAP computing
            outputs_array = outputs.clone().cpu()
            outputs_array = F.softmax(outputs_array, dim = 1)
            outputs_array = outputs_array.detach().numpy()
            maps = np.append(maps, outputs_array, axis = 0)
            labels_array = labels.clone().cpu()
            labels_array = labels_array.detach().numpy()
            lab = np.append(lab, labels_array)
            
            # Model predictions 
            _, preds = torch.max(outputs, 1)

            # backward + optimize
            loss.backward()
            optimizer.step()
        
        # Statistics

        # Loss
        running_loss += loss.item() * inputs.size(0)

        # Accuracy
        running_corrects += torch.sum(preds == labels.data)

    # Loss
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Accuracy
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    # NAP
    epoch_nap, epoch_f = norm_ap_optimized(maps,lab, num_classes)                
    
    # AUC
    epoch_auc = pltauc(maps, lab, num_classes)
        
    # Print in screen
    print('Phase Train, Loss: {:.4f} Acc: {:.4f} NAP: {:.4f} F-measure: {:.4f} auc:  {:.4f}'.format(epoch_loss,
                                                                                                    epoch_acc,
                                                                                                    epoch_nap[-1],
                                                                                                    epoch_f[-1],
                                                                                                    epoch_auc["micro"]))
    
    return epoch_loss, epoch_acc, epoch_nap[-1], epoch_f[-1], epoch_auc["micro"]


def evaluate(epoch):
    net.eval()

    # Epoch Scores and loss
    running_loss = 0.0
    running_corrects = 0.0

    lab =  np.array([])
    maps = np.empty((0, num_classes))

    # Iterate over data
    for inputs, _, labels in test_loader:

        inputs  = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward

        # track history if only in train
        with torch.set_grad_enabled(False):

            # Get model outputs and calculate loss
            outputs= net(inputs)
            loss = 0
            for i in range(num_classes):
                labe = labels.clone()
                labe[labels == i] = 1
                labe[labels != i] = 0
                loss += criterion(F.sigmoid(outputs[:,i].float()),
                                  labe.float().to(device))

            # Output modification for mAP computing
            outputs_array = outputs.clone().cpu()
            outputs_array = F.softmax(outputs_array, dim = 1)
            outputs_array = outputs_array.detach().numpy()
            maps = np.append(maps, outputs_array, axis = 0)
            labels_array = labels.clone().cpu()
            labels_array = labels_array.detach().numpy()
            lab = np.append(lab, labels_array)
            
            # Model predictions 
            _, preds = torch.max(outputs, 1)

        # Statistics

        # Loss
        running_loss += loss.item() * inputs.size(0)
       
        # Accuracy
        running_corrects += torch.sum(preds == labels.data)

    # Scheduler
    scheduler.step(loss)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        
    # Loss
    epoch_loss = running_loss / len(test_loader.dataset)
    
    # Accuracy
    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    # NAP
    epoch_nap, epoch_f = norm_ap_optimized(maps,lab, num_classes)                
    
    # AUC
    epoch_auc = pltauc(maps, lab, num_classes)
    
    predictions = [maps, lab]
    
    # Print in screen
    print('Phase Validation, Loss: {:.4f} Acc: {:.4f} NAP: {:.4f} F-measure: {:.4f} auc:  {:.4f}'.format(epoch_loss,
                                                                                                       epoch_acc,
                                                                                                       epoch_nap[-1],
                                                                                                       epoch_f[-1],
                                                                                                       epoch_auc["micro"]))
    
    return epoch_loss, epoch_acc, epoch_nap[-1], epoch_f[-1], epoch_auc["micro"], predictions

# -------------------- TRAIN AND TEST FUNCTIONS -------------------
     

initial_epoch = 0
num_epochs = args["epochs"]

train_losses_history = []
val_losses_history = []
# Accuracy
train_acc_history = []
val_acc_history = []
# NAP 
train_nap_history = []
val_nap_history = []
# F measure 
train_f_history = []
val_f_history = []
# AUC
train_auc_history = []
val_auc_history = []

best_model_wts = copy.deepcopy(net.state_dict())

saving_path_models = os.path.join(saving_path,'model'+str(cross_val)+'/')
if not os.path.exists(saving_path_models):
    os.makedirs(saving_path_models,0o777)


if args["checkpoint"]: 
    model_checkpoint = torch.load(os.path.join(saving_path_models, 'Checkpoint.pth'))
    net.load_state_dict(model_checkpoint['model_state_dict'])
    optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
    initial_epoch = model_checkpoint['epoch']
    train_losses_history = model_checkpoint['loss']


def main():
    since = time.time()
    best_NAP = 0

    if args["checkpoint"]:
        val_loss, val_acc, val_nap, val_f, val_auc, prediction_nap = evaluate(initial_epoch)
        best_NAP = val_nap

    for epoch in range(initial_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #---------------------- Train  ----------------------------
        train_loss, train_acc, train_nap, train_f, train_auc = train(epoch)        
        train_losses_history.append(train_loss)
        # Accuracy
        train_acc_history.append(train_acc)
        # AUC
        train_auc_history.append(train_auc)
        # NAP 
        train_nap_history.append(train_nap)
        # F measure
        train_f_history.append(train_f)
        
        #----------------------- Validation -----------------------
        val_loss, val_acc, val_nap, val_f, val_auc, prediction_nap = evaluate(epoch)
        val_losses_history.append(val_loss)
        # Accuracy
        val_acc_history.append(val_acc)
        # AUC
        val_auc_history.append(val_auc)
        # NAP
        val_nap_history.append(val_nap)
        # F measure
        val_f_history.append(val_f)

        if val_nap > best_NAP:
                best_NAP = val_nap
                best_prediction = prediction_nap
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_losses_history
                            }, os.path.join(saving_path_models, 'Checkpoint.pth'))
            
        plot_loss(train_losses_history,
                val_losses_history, 
                save_dir= saving_path_models)
        plot_nap(val_nap_history, val_auc_history, 
                torch.tensor(val_acc_history).cpu().tolist(),
                train_nap_history, train_auc_history, 
                torch.tensor(train_acc_history).cpu().tolist(),
                save_dir= saving_path_models)    

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val NAP: {:4f}'.format(best_NAP))
    print('Best val ACC: {:4f}'.format(max(val_acc_history)))
    print('Best val AUC: {:4f}'.format(max(val_auc_history)))
    print('Best val Fscore: {:4f}'.format(max(val_f_history)))
    # Save the model weights
    net.load_state_dict(best_model_wts)

    
    torch.save({
                'model' : net.state_dict(),
                'optimize' : optimizer.state_dict(),
                'prediction': best_prediction,
                'charset': char_to_int,
                'embed': embed
                }, os.path.join(saving_path_models, 'model.pth'))

    # Save the experiment configuration

    save_items = args.copy()
    del save_items["train_path"]
    del save_items["test_file"]

    save_items["Date"] = datetime.date.today()
    save_items["Best_NAP"] = best_NAP
    save_items["Best_ACC"] = max(val_acc_history)
    save_items["Best_AUC"] = max(val_auc_history)
    save_items["Best_Fscore"] = max(val_f_history)

    fieldnames = list(save_items.keys())

    csv_file = os.path.join(args['results_dir'], 'Results.csv')

    if os.path.exists(csv_file):

        with open(csv_file, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames= fieldnames)
                writer.writerow(save_items)
                
    else:
        
        with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames= fieldnames)
                writer.writeheader()
                writer.writerow(save_items)
        

if __name__ == '__main__':

    main()
