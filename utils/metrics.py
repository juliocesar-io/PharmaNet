from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')
import numpy as np

from itertools import cycle
from sklearn.utils.extmath import stable_cumsum
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (precision_recall_curve,
                             average_precision_score,
                             roc_curve,
                             roc_auc_score, auc)

import argparse
import numpy as np

def norm_ap_optimized(output,target, num_classes = 102):

    F1_T = []
    N_total = len(output)/(max(target)+1)
    area_t = []
    for clas in range(0,num_classes):
        area = 0
        y_true = (target==clas).astype(int)
        y_score = output[:,clas]
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        TP = stable_cumsum(y_true)[threshold_idxs]  
        FP = stable_cumsum((1 - y_true))[threshold_idxs]
        TN = [FP[-1] for x in range(0,len(FP))]
        TN = np.subtract(TN,FP)
        FN = [TP[-1] for x in range(0,len(TP))]
        FN = np.subtract(FN,TP)  
        
        # Recall   
        Recall = np.divide(TP,np.add(TP,FN))
        
        # Normalized Precision
        Precision = np.divide(
                             np.multiply(Recall,N_total),
                             np.add(np.multiply(
                                    Recall,N_total),FP))
        
        denom = np.add(Precision,Recall)
        denom[denom == 0] = 1
        
        # F-measure
        F1= np.divide(np.multiply(2,np.multiply(Precision,Recall)),denom)        
        
        # Compute Area under Normalized Precision Recall curve
        for i,esto in enumerate(zip(np.array(Recall)[:-1],np.array(Recall)[1:])): 
            area+=(esto[1]-esto[0])*Precision[i]
        F1_T.append(np.max(F1))
        area_t.append(area)
    
    # Compute Area under NAP curve
    area_under_curve = [0,1]
    area_under_curve += area_t
    area_under_curve = sorted(area_under_curve)
    nap_area = 0.
    scores = np.arange(num_classes, -1, -1)
    score = np.insert(scores, 0, num_classes)

    for i,esto in enumerate(zip(np.array(area_under_curve)[:-1],
                                np.array(area_under_curve)[1:])): 
        nap_area+=(esto[1]-esto[0])*score[i]

    # For same size 
    F1_T.append(np.mean(F1_T))

    # Final value for area under NAP curve divided by the number of classes.
    area_t.append(nap_area/num_classes)

    return area_t, F1_T

def norm_ap(output,target, num_classes = 102):

    F1_T = []
    N_total = len(output)/(max(target)+1)
    area_t = []

    for clas in range(0,num_classes):
        area = 0
        R_total = []
        P_total = []
        F1_total = []
        for thr in np.arange(0,1,0.0001):
            groundtruth = (target==clas).astype(int)
            predicted_thr = (output[:,clas]>=thr).astype(int)
            TP = np.sum((predicted_thr == 1) & (groundtruth ==1))
            FN = np.sum((predicted_thr == 0) & (groundtruth ==1))
            FP = np.sum((predicted_thr == 1) & (groundtruth ==0))
            
            # Recall   
            Recall = TP/(TP+FN)
            
            # Normalized Precision
            
            Precision = (Recall * N_total)/((Recall*N_total)+FP)
            if np.isnan(Precision):
                Precision = 0
            # F-measure
            denom = Precision + Recall
            if denom == 0:
                denom = 1
            F1= 2*(Precision*Recall)/denom
            R_total.append(Recall)
            P_total.append(Precision)
            F1_total.append(F1)
        # Compute Area under Normalized Precision Recall curve
        for i,esto in enumerate(zip(np.array(R_total)[:-1],np.array(R_total)[1:])): 
            area+=(esto[0]-esto[1])*P_total[i]
        
        # F-measure
        F1_T.append(np.max(F1_total))
        area_t.append(area)
   
    # Compute Area under NAP curve
    area_under_curve = [0,1]
    area_under_curve += area_t
    area_under_curve = sorted(area_under_curve)
    nap_area = 0.
    scores = np.arange(num_classes, -1, -1)
    score = np.insert(scores, 0, num_classes)

    for i,esto in enumerate(zip(np.array(area_under_curve)[:-1],
                                np.array(area_under_curve)[1:])): 
        nap_area+=(esto[1]-esto[0])*score[i]
    
    F1_T.append(np.mean(F1_T))

    # Final value for area under NAP curve divided by the number of classes.
    area_t.append(nap_area/num_classes)

    return area_t, F1_T

def pltmap(output, target, num_classes):

    new_labels = label_binarize(target, classes= list(range(0,num_classes)))
    n_classes = new_labels.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    fmeasure = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(new_labels[:, i], output[:, i])
        average_precision[i] = average_precision_score(new_labels[:, i], output[:, i])
        denom = precision[i]+recall[i] 
        denom[denom == 0.] = 1
        fmeasure[i] = np.max(2*precision[i]*recall[i] / denom)
         

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(new_labels.ravel(), output.ravel())
    average_precision["micro"] = average_precision_score(new_labels, output, average="macro")
    denom = precision["micro"]+recall["micro"] 
    denom[denom == 0.] = 1
    fmeasure["micro"] = np.max(2*precision["micro"]*recall["micro"] / denom)
    

    return average_precision, fmeasure

def pltauc(output, target, num_classes):

    new_labels = label_binarize(target, classes= list(range(0,num_classes)))
    n_classes = new_labels.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
   
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(new_labels[:, i], output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # A "micro-average": quantifying score on all classes jointly
    fpr["micro"], tpr["micro"], _ = roc_curve(new_labels.ravel(), output.ravel())
    roc_auc["micro"] = roc_auc_score(new_labels, output, average="macro")
        
    return roc_auc
