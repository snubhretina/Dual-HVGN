from __future__ import print_function

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.autograd import Variable
from datetime import datetime
import csv
from scipy.ndimage import rotate
from model import *
from collections import OrderedDict

# select gpu number
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Deep Vessel')
parser.add_argument('--batch', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--test-batch', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

save_eval_path = 'test_inform/'
if os.path.exists(save_eval_path) == False:
    os.mkdir(save_eval_path)

save_res_path = 'test_res/'
if os.path.exists(save_res_path) == False:
    os.mkdir(save_res_path)

def calc_pr_auc(gt, pred):
    precision, recall, threshold = precision_recall_curve(gt, pred)
    pr_auc = auc(recall, precision)
    return precision, recall, threshold, pr_auc
def calc_roc_auc(gt, pred):
    fpr, tpr, _ = roc_curve(gt, pred, pos_label=1.)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# if args.cuda:
#     torch.cuda.manual_seed(100)

te_b_size = args.test_batch

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

nY = 1024
nX = 1536

# load train/test path
db_path = '/mnt/data4/data/02_SNUBH_FIREFLY/01_1st_2nd_449set/'
tr_path = '../../../../00_DB_list/tr_list_359set.csv'
te_path = '../../../../00_DB_list/te_list_90set.csv'

ftr = open(tr_path)
fte = open(te_path)

fr = csv.reader(ftr)
fe = csv.reader(fte)

train_data_path= []
test_data_path =  []

for i in fr:
    train_data_path.append(i[0])
for j in fe:
    test_data_path.append(j[0])

train_data_path = sorted(train_data_path) 
test_data_path = sorted(test_data_path)

def load_te_img(idx):
    data = torch.FloatTensor(te_b_size, 4, nY, nX)
    label = torch.FloatTensor(te_b_size, 2, nY, nX)

    im = Image.open(db_path + 'fundus/' + test_data_path[idx])
    gt =  Image.open(db_path + 'mask/' + test_data_path[idx])

    im = np.array(im, dtype=np.float32)
    gt = np.array(gt)

    numpy_data = np.zeros([4, nY, nX], dtype=np.float32)
    numpy_label = np.zeros([2, nY, nX], dtype=np.float32)

    seg = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    seg[np.bitwise_or(gt[:,:,0] != 0, gt[:,:,2] != 0)] = 1

    numpy_data[0] = seg
    numpy_data[1] = im[:,:,0]
    numpy_data[2] = im[:,:,1]
    numpy_data[3] = im[:,:,2]

    label_A = np.zeros((nY, nX), dtype=np.uint8)
    label_V = np.zeros((nY, nX), dtype=np.uint8)

    artery = np.bitwise_and(gt[:,:,0]!=0, np.bitwise_and(gt[:,:,1]==0, gt[:,:,2]== 0))
    vein = np.bitwise_and(gt[:,:,2]!=0, np.bitwise_and(gt[:,:,0]==0, gt[:,:,1]==0))
    crossing = np.bitwise_and(gt[:,:,1]==0, np.bitwise_and(gt[:,:,0]!=0, gt[:,:,2]!=0))

    label_A[artery != 0] = 1
    label_A[crossing !=0 ] = 1
    label_A[vein] = 2
    label_A[numpy_data[0] == 0] = 0

    label_V[vein != 0] = 2
    label_V[crossing != 0] = 2
    label_V[artery != 0] = 1
    label_V[numpy_data[0] == 0] = 0

    numpy_label[0] = label_A
    numpy_label[1] = label_V
    
    for j in range(numpy_data.shape[0]):
        if j >= 1:
            numpy_data[j] -= np.mean(numpy_data[j, numpy_data[0] != 0])
        
    numpy_data = torch.from_numpy(numpy_data)
    numpy_label = torch.from_numpy(numpy_label)

    data[0] = numpy_data
    label[0][0] = numpy_label[0]
    label[0][1] = numpy_label[1]

    return data, label

cnn_pr_path = 'model_save/cnn_model_12000_iter_loss_0.1586_acc_0.9330.pth.tar'
gnn_pr_path = 'model_save/gnn_model_12000_iter_loss_0.1586_acc_0.9330.pth.tar'

# model - SSANet + GUNet        
cnn = SSANet().cuda(1)
cnn.load_state_dict(torch.load(cnn_pr_path, map_location='cuda:1'))

# gnn_state = torch.load(gnn_pr_path, map_location='cuda:0')

# gnn_A_dict = OrderedDict()
# gnn_V_dict = OrderedDict()

# for i, param in gnn_state.items():
#     if i.find('GUnet_A') != -1:
#         gnn_A_dict[i[8:]] = param
#     else:
#         gnn_V_dict[i[8:]] = param

graph_unet = GNN().cuda(1)
graph_unet.load_state_dict(torch.load(gnn_pr_path, map_location='cuda:1'))

# segmentation loss
criterion_bce = nn.BCELoss().cuda(1)

def test(test_loss_list):
    cnn.eval()
    graph_unet.eval()

    t = 0
    iter = len(test_data_path)
    pred_list = []
    gt_list = []

    with torch.no_grad():
        for i in range(iter):
            st = datetime.now()

            data, label = load_te_img(i)
            gnn_label_A = label[0, 0, data[0, 0]!=0] -1
            mt_gnn_label_A = torch.zeros([gnn_label_A.size(0), 2])
            mt_gnn_label_A[gnn_label_A == 0, 0] = 1
            mt_gnn_label_A[gnn_label_A == 1, 1] = 1

            gnn_label_V = label[0, 1, data[0, 0]!=0] -1
            mt_gnn_label_V = torch.zeros([gnn_label_V.size(0), 2])
            mt_gnn_label_V[gnn_label_V == 0, 0] = 1
            mt_gnn_label_V[gnn_label_V == 1, 1] = 1

            data, label = Variable(data).cuda(1), Variable(label).cuda(1)
            mt_gnn_label_A, mt_gnn_label_V = Variable(mt_gnn_label_A).cuda(1), Variable(mt_gnn_label_V).cuda(1)

            # CNN
            feature, _, _ = cnn(data)
    
            # Graph-UNet
            outputs = graph_unet(feature, data, label)

            # GNN Loss
            gnn_Aloss = criterion_bce(outputs[0], mt_gnn_label_A)
            gnn_Vloss = criterion_bce(outputs[1], mt_gnn_label_V)

            gnn_loss = 0.5 * (gnn_Aloss + gnn_Vloss)

            test_loss_list.append(gnn_loss.item())
            
            et = datetime.now()
            t += (et-st).total_seconds()
            
            for output in outputs:
                pred = output.view(-1).data.cpu().numpy().copy()
                pred_list = pred_list + [v for v in pred]
            for mt_gnn_label in [mt_gnn_label_A, mt_gnn_label_V]:
                label_numpy = mt_gnn_label.view(-1).data.cpu().numpy().copy()
                gt_list = gt_list + [v for v in label_numpy]
            nonzero_axis = np.nonzero(data[0][0])
            canvas_output = np.zeros([nY, nX, 3], dtype=np.ubyte)
            for idx, gnn_label_idx in enumerate([gnn_label_A, gnn_label_V]):
                _, pred = F.softmax(outputs[idx], dim=1).max(dim=1)
                for j in range(gnn_label_idx.size(0)):
                    # if pred[j].item() == idx:
                    # canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], idx*2] = 255
                    if idx == 0:
                        if pred[j].item() == idx:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], idx] = 255
                        else:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], idx+2] = 255                        
                    else:
                        if pred[j].item() == idx:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], idx*2] = 255    
                        else:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], idx-1] = 255                            
                        
            output_save_path = save_res_path + '%05d.png' % (i)
            Image.fromarray(canvas_output).save(output_save_path)

    pred_arr = np.array(pred_list).flatten()
    gt_arr = np.array(gt_list).flatten()

    precision, recall, threshold, pr_auc_score = calc_pr_auc(gt_arr, pred_arr)
    fpr, tpr, roc_auc_score = calc_roc_auc(gt_arr, pred_arr)

    all_f1 = 2. * precision * recall / (precision + recall)
    best_f1 = np.nanmax(all_f1)
    index = np.nanargmax(all_f1)
    best_f1_threshold = threshold[index]
    binary_flat = (pred_arr >= best_f1_threshold).astype(np.float32)
    acc = (gt_arr == binary_flat).sum() / float(pred_arr.shape[0])

    tp = np.bitwise_and((gt_arr == 1).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
    tn = np.bitwise_and((gt_arr == 0).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
    fp = np.bitwise_and((gt_arr == 0).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
    fn = np.bitwise_and((gt_arr == 1).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
    se = tp / float(tp + fn)
    sp = tn / float(fp + tn)

    score = [pr_auc_score, roc_auc_score, best_f1, best_f1_threshold, acc, se, sp]

    return  test_loss_list, iter, t, score


test_loss_list = []

# test()
# score : [pr_auc, roc_auc, best_f1, acc, best_f1_threshold, se, sp]
test_loss_list, test_iter, test_t, score = test(test_loss_list)

f = open(save_eval_path + 'all_test_result.csv', 'w')
csv_file = csv.writer(f)
csv_file.writerow(
    ['PR', 'ROC', 'F1', 'thresh', 'ACC', 'SE', 'SP'])

csv_file.writerow(score)
f.close()

