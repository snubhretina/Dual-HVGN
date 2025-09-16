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

save_dir_path = 'model_save/'
if os.path.exists(save_dir_path) == False:
    os.mkdir(save_dir_path)

save_eval_path = 'inform/'
if os.path.exists(save_eval_path) == False:
    os.mkdir(save_eval_path)

save_res_path = 'res/'
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

if args.cuda:
    torch.cuda.manual_seed(100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_iter = 50000
log_itv = 10
log_test = 200
b_size = args.batch
te_b_size = args.test_batch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

nY = 1024
nX = 1536

# load train/test path
db_path = '/mnt/data4/data/02_SNUBH_FIREFLY/01_1st_2nd_449set/'
tr_path = '../../../../00_DB_list/tr_list_359set.csv'
te_path = '../../../../00_DB_list/te_list_90set.csv'
cnn_pr_path = '../../../../00_DB_list/01_pretrained_model/01_CNN_GNN/cnn_model_25000_iter_loss_0.1585_acc_0.9330.pth.tar'
gnn_pr_path = '../../../../00_DB_list/01_pretrained_model/01_CNN_GNN/gnn_model_25000_iter_loss_0.1585_acc_0.9330.pth.tar'

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

def load_tr_img():
    randp = np.random.randint(0, len(train_data_path) - 1, b_size)
    data = torch.FloatTensor(b_size, 4, nY, nX)
    label = torch.LongTensor(b_size, 2, nY, nX)

    for i in range(b_size):

        im = Image.open(db_path + 'fundus/' + train_data_path[randp[i]])
        gt = Image.open(db_path + 'mask/' + train_data_path[randp[i]])

        im = np.array(im, dtype=np.float32)
        gt = np.array(gt)

        numpy_data = np.zeros([4, nY, nX], dtype=np.float32)
        numpy_label = np.zeros([2, nY, nX], dtype=np.float32)

        seg = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
        seg[np.bitwise_or(gt[:,:,0] != 0, gt[:,:,2] != 0)] = 1

        if np.random.random_sample(1) > 0.5:
            im = im[:,::-1].copy()
            gt = gt[:,::-1].copy()
            seg = seg[:,::-1].copy()

        im += np.random.uniform(-0.3, 0.3) * 255
        im = np.clip(im, 0, 255)

        mmR = np.mean(im[:,:,0])
        mmG = np.mean(im[:,:,1])
        mmB = np.mean(im[:,:,2])
        rand_contrast = np.random.uniform(0.5,1.5,1)

        im[:,:,0] = (im[:,:,0] - mmR) * rand_contrast[0] + mmR
        im[:,:,1] = (im[:,:,1] - mmG) * rand_contrast[0] + mmG
        im[:,:,2] = (im[:,:,2] - mmB) * rand_contrast[0] + mmB
        im = np.clip(im, 0, 255)

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
        
        # Data augmentation - rotation
        angle = np.random.randint(-30,30,1)
        for j in range(numpy_data.shape[0]):
            numpy_data[j] = Image.fromarray(numpy_data[j]).rotate(angle)
            if j >= 1:
                numpy_data[j] -= np.mean(numpy_data[j, numpy_data[0] != 0])

        for j in range(numpy_label.shape[0]):
            numpy_label[j] = np.array(Image.fromarray(numpy_label[j]).rotate(angle), dtype=np.float32)

        numpy_data = torch.from_numpy(numpy_data)
        numpy_label = torch.from_numpy(numpy_label)
    
        data[i] = numpy_data
        label[i][0] = numpy_label[0]
        label[i][1] = numpy_label[1]
        
    return data, label


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

# model - SSANet + GUNet        
cnn = SSANet().cuda(0)
cnn.load_state_dict(torch.load(cnn_pr_path, map_location='cuda:0'))

cnt = 0
for i, chil in enumerate(cnn.children()):
    cnt += 1
    if cnt < 13:
        for param in chil.parameters():
            param.requires_grad = False

graph_unet = GNN().cuda(0)
graph_unet.load_state_dict(torch.load(gnn_pr_path, map_location='cuda:0'))
# initialize optimizer
optimizer = optim.Adam(graph_unet.parameters(), lr=args.lr, betas=(0.5, 0.999))

# segmentation loss
criterion_bce = nn.BCELoss().cuda(0)

def train(t, train_loss_list):
    st = datetime.now()
    cnn.train()
    graph_unet.train()

    # data - 4 channel(0:segm, 1-3: FP)
    # label - 2 channel(0:artery(up)/vein, 1:artery/vein(up))
    # 0:bg, 1:artery, 2:vein
    data, label = load_tr_img()

    # CNN label -> GNN label
    gnn_label_A = label[0, 0, data[0, 0]!=0] -1
    mt_gnn_label_A = torch.zeros([gnn_label_A.size(0), 2])
    mt_gnn_label_A[gnn_label_A == 0, 0] = 1
    mt_gnn_label_A[gnn_label_A == 1, 1] = 1

    gnn_label_V = label[0, 1, data[0, 0]!=0] -1
    mt_gnn_label_V = torch.zeros([gnn_label_V.size(0), 2])
    mt_gnn_label_V[gnn_label_V == 0, 0] = 1
    mt_gnn_label_V[gnn_label_V == 1, 1] = 1

    data, label = Variable(data).cuda(0), Variable(label).cuda(0)
    mt_gnn_label_A, mt_gnn_label_V = Variable(mt_gnn_label_A).cuda(0), Variable(mt_gnn_label_V).cuda(0)

    optimizer.zero_grad()
    
    # freezing CNN
    feature, _, _ = cnn(data)
    
    # Graph-UNet
    outputs = graph_unet(feature, data, label)

    # GNN Loss
    gnn_Aloss = criterion_bce(outputs[0], mt_gnn_label_A)
    gnn_Vloss = criterion_bce(outputs[1], mt_gnn_label_V)

    gnn_loss = 0.5 * (gnn_Aloss + gnn_Vloss)
    
    gnn_loss.backward()
    optimizer.step()
    scheduler.step()

    et = datetime.now()
    t += (et-st).total_seconds()

    train_loss_list.append(gnn_loss.item())
    
    return t, train_loss_list, gnn_loss.item()

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

            data, label = Variable(data).cuda(0), Variable(label).cuda(0)
            mt_gnn_label_A, mt_gnn_label_V = Variable(mt_gnn_label_A).cuda(0), Variable(mt_gnn_label_V).cuda(0)

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

train_t = 0
mean_train_loss_list = []
mean_test_loss_list = []
train_loss_list = []
test_loss_list = []
test_acc_list = []
test_pr_list = []
test_roc_list = []
test_f1_list = []
test_se_list = []
test_sp_list = []
all_test_scores = []

# set learning-rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

for iter in range(max_iter+1):
    cur_lr = optimizer.param_groups[0]['lr']

    # train()
    train_t, train_loss_list, cur_train_loss  = train(train_t,train_loss_list)

    if (iter + 1) % log_itv == 0:
        print('Train iter: %d [%d/%d (%.0f)] Loss: %.4f, time per frame: %.4f, total time: %.4f(bs:%d)' \
              % (max_iter, (iter + 1), max_iter,
                 100. * (iter + 1) / max_iter, cur_train_loss, train_t / float(log_itv) / float(b_size),
                 train_t, b_size))    
        train_t = 0

    if (iter+1) % log_test == 0:
        print('\ncur dir: %s, rl: %e' % (os.getcwd(), cur_lr))
        # test()
        # score : [pr_auc, roc_auc, best_f1, acc, best_f1_threshold, se, sp]
        test_loss_list, test_iter, test_t, score = test(test_loss_list)

        print('\nTest set:time per frame: %.4f, total time: %.4f(%d)\n' % (
             test_t / float(test_iter * b_size), test_t, test_iter * b_size))

        mean_train_loss_list.append(np.mean(train_loss_list))
        mean_test_loss_list.append(np.mean(test_loss_list))
        test_pr_list.append(score[0])
        test_roc_list.append(score[1])
        test_f1_list.append(score[2])
        test_acc_list.append(score[4])
        test_se_list.append(score[5])
        test_sp_list.append(score[6])
        all_test_scores.append(score)
        
        save_path = save_dir_path + 'cnn_model_%d_iter_loss_%.4f_acc_%.4f.pth.tar' % (
            iter + 1, mean_test_loss_list[-1], test_acc_list[-1])
        torch.save(cnn.state_dict(), save_path)
        
        save_path = save_dir_path + 'gnn_model_%d_iter_loss_%.4f_acc_%.4f.pth.tar' % (
            iter + 1, mean_test_loss_list[-1], test_acc_list[-1])
        torch.save(graph_unet.state_dict(), save_path)

        # draw the result
        cnt = len(mean_train_loss_list)
        lw = 1
        x_range = range(log_test, log_test * (cnt + 1), log_test)
        fig = plt.figure(figsize=(10, 8))
        plt.plot(x_range, np.array(mean_train_loss_list), 'g', lw=lw, label='train loss')
        plt.plot(x_range, np.array(mean_test_loss_list), 'b', lw=lw, label='test loss')
        plt.grid(True)
        plt.title('train-test loss')
        plt.legend(loc="upper right")
        fig.savefig(save_eval_path + '1_loss.png')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(x_range, np.array(test_roc_list), 'g', lw=lw, label='ROC')
        plt.plot(x_range, np.array(test_pr_list), 'b', lw=lw, label='PR')
        plt.grid(True)
        plt.title('score')
        plt.legend(loc="lower right")
        fig.savefig(save_eval_path + '2_PR_ROC.png')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(x_range, np.array(test_acc_list), 'g', lw=lw, label='ACC')
        plt.plot(x_range, np.array(test_f1_list), 'b', lw=lw, label='F1')
        plt.plot(x_range, np.array(test_se_list), lw=lw, label='SE')
        plt.plot(x_range, np.array(test_sp_list), lw=lw, label='SP')
        plt.grid(True)
        plt.title('score')
        plt.legend(loc="lower right")
        fig.savefig(save_eval_path + '3_ACC_F1_SE_SP.png')
        plt.close()

        f = open(save_eval_path + 'all_test_result.csv', 'w')
        csv_file = csv.writer(f)
        csv_file.writerow(
            ['iter', 'train_loss', 'test_loss', 'PR', 'ROC', 'F1', 'thresh', 'ACC', 'SE', 'SP'])
        for i in range(mean_train_loss_list.__len__()):
            csv_file.writerow([x_range[i], mean_train_loss_list[i], mean_test_loss_list[i]] + all_test_scores[i])
        f.close()

        train_loss_list = []
        test_loss_list = []
