
import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from layers import SNNModel
# from keras.models import load_model, Model
# from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout
# from keras.callbacks import EarlyStopping
# from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import datetime

MAX_LEN = 50

def one_hot_padding(seq_list,padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences, 
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    pos, neg, neut_charge=['K', 'R'], ['D', 'E'], ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    hpho, pol, neut_hydro=['F', 'I', 'W', 'C'], ['K', 'D', 'E', 'Q', 'P', 'S', 'R', 'N', 'T', 'G'], ['A', 'H', 'Y', 'M', 'L', 'V']
    g1_p , g2_p, g3_p = ['G', 'A', 'S', 'D', 'T'], ['C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'], ['K', 'M', 'H', 'F', 'R', 'Y', 'W']
    g1_v, g2_v, g3_v = ['G', 'A', 'S', 'T', 'P', 'D'], ['N', 'V', 'E', 'Q', 'I', 'L'], ['M', 'H', 'K', 'F', 'R', 'Y', 'W']
    H,S, C =['E','A', 'L', 'M', 'Q', 'K', 'R', 'H'], ['V','I','Y','C','W','F','T'], ['G', 'N', 'P', 'S', 'D']

    
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*35
        one_hot[aa[i]][i] = 1 
        if aa[i] in pos:
            one_hot[aa[i]][20]=1
        elif aa[i] in neg:
            one_hot[aa[i]][21]=1
        elif aa[i] in neut_charge:
            one_hot[aa[i]][22]=1
        
        if aa[i] in hpho:
            one_hot[aa[i]][23]=1
        elif aa[i] in pol:
            one_hot[aa[i]][24]=1
        elif aa[i] in neut_hydro:
            one_hot[aa[i]][25]=1

        if aa[i] in H:
            one_hot[aa[i]][26]=1
        elif aa[i] in S:
            one_hot[aa[i]][27]=1
        elif aa[i] in C:
            one_hot[aa[i]][28]=1

        if aa[i] in g1_p:
            one_hot[aa[i]][29]=1
        elif aa[i] in g2_p:
            one_hot[aa[i]][30]=1
        elif aa[i] in g3_p:
            one_hot[aa[i]][31]=1

        if aa[i] in g1_v:
            one_hot[aa[i]][32]=1
        elif aa[i] in g2_v:
            one_hot[aa[i]][33]=1
        elif aa[i] in g3_v:
            one_hot[aa[i]][34]=1
        
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*35]*(padding-len(seq_list[i]))
        feat_list.append(feat)   
        
    feat_list=torch.from_numpy(np.array(feat_list))

    return feat_list

def predict_by_class(scores):
    """
    Turn prediction scores into classes.
    If score > 0.5, label the sample as 1; else 0.
    Input: scores - scores predicted by the model, 1-d array.
    Output: an array of 0s and 1s.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append(1)
        else:
            classes.append(0)
    return torch.from_numpy(np.array(classes))

def main():
    parser = argparse.ArgumentParser(description=dedent('''
        AMPlify v1.1.0 training
        ------------------------------------------------------
        Given training sets with two labels: AMP and non-AMP,
        train the AMP prediction model.    
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # parser.add_argument('-amp_tr', help="Training AMP set, fasta file", required=True)
    # parser.add_argument('-non_amp_tr', help="Training non-AMP set, fasta file", required=True)
    parser.add_argument('-amp_te', help="Test AMP set, fasta file (optional)", default=None, required=True)
    parser.add_argument('-non_amp_te', help="Test non-AMP set, fasta file (optional)", default=None, required=True)
    # parser.add_argument('-sample_ratio', 
    #                     help="Whether the training set is balanced or not (balanced by default, optional)", 
    #                     choices=['balanced', 'imbalanced'], default='balanced', required=False)
    # parser.add_argument('-out_dir', help="Output directory", required=False)
    # parser.add_argument('-model_name', help="File name of trained model weights", required=False)
    
    args = parser.parse_args()

    # load test sets
    AMP_test = []
    non_AMP_test = []      
    for seq_record in SeqIO.parse(args.amp_te, 'fasta'):
        # "../data/AMPlify_AMP_test_common.fa"
        AMP_test.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_te, 'fasta'):
        # "../data/AMPlify_non_AMP_test_balanced.fa"
        non_AMP_test.append(str(seq_record.seq))

    # sequences for test sets
    test_seq = AMP_test + non_AMP_test
    # set labels for test sequences
    y_test = torch.from_numpy(np.array([1]*len(AMP_test) + [0]*len(non_AMP_test)))
    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_test = one_hot_padding(test_seq, MAX_LEN)
    indv_pred_test = []

    in_channels = 64
    heads=2
    out_channels=32
    model = SNNModel(X_test.shape, in_channels, heads, out_channels)
    # model.test()
    model.load_state_dict(torch.load('Models/Model_weight_4.h5'))

    temp_pred_test = torch.squeeze(model(X_test.float())).detach().numpy()# predicted scores on the test set from the current model
    indv_pred_test.append(temp_pred_test)
    temp_pred_class_test = predict_by_class(temp_pred_test)
    tn_indv, fp_indv, fn_indv, tp_indv = confusion_matrix(y_test, temp_pred_class_test).ravel()
    import pdb;pdb.set_trace()
    #print(confusion_matrix(y_test, temp_pred_class_test))
    print()

    print('test acc: ', accuracy_score(y_test, temp_pred_class_test))  
    print('test sens: ', tp_indv/(tp_indv+fn_indv))
    print('test spec: ', tn_indv/(tn_indv+fp_indv))
    print('test f1: ', f1_score(y_test, temp_pred_class_test))
    print('test roc_auc: ', roc_auc_score(y_test, temp_pred_test))

if __name__ == "__main__":
    main()