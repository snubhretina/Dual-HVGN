import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = self.downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SSANet(nn.Module):
    def __init__(self):
        super(SSANet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 1 , 3, bias=False)
        # self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = self._make_layer(BasicBlock, 64, 3, stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4, stride=2)
        # self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6, stride=2)
        # self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3, stride=2)
        # self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp2 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp3 = nn.Sequential(
            nn.Conv2d(128, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp4 = nn.Sequential(
            nn.Conv2d(256, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp5 = nn.Sequential(
            nn.Conv2d(512, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        # self.outputA = self.make_infer(2, 1+5*16))
        # self.outputV = self.make_infer(2, 1+5*16))
        self.classifier_A = nn.Sequential(nn.Conv2d(1+16*5, 1, 1, 1, 0))
        self.classifier_V = nn.Sequential(nn.Conv2d(1+16*5, 1, 1, 1, 0))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 16, 3, 1, 1),
                    # nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(16, 16, 3, 1, 1),
                    # nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True)))
            infer_layers.append(nn.Sequential(nn.Conv2d(16, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):
        c1 = self.conv1(x[:, 1:4, :, :])
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = self.sp1(c1)
        c1 = F.interpolate(c1, size=(x.size(2) // 2, x.size(3)//2), mode='bilinear', align_corners=True)

        c2 = self.conv2(c1)
        sp2 = F.interpolate(self.sp2(c2), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        # c2 = F.interpolate(c2, size=(x.size(2)//2, x.size(3)//2), mode='bilinear', align_corners=True)

        c3 = self.conv3(c2)
        sp3 = F.interpolate(self.sp3(c3), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        # c3 = F.interpolate(c3, size=(x.size(2)//2, x.size(3)//2), mode='bilinear', align_corners=True)

        c4 = self.conv4(c3)
        sp4 = F.interpolate(self.sp4(c4), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        # c4 = F.upsample(c4, size=(x.size(2)/2, x.size(3)/2), mode='bilinear', align_corners=True)

        c5 = self.conv5(c4)
        sp5 = F.interpolate(self.sp5(c5), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        cat = torch.cat([x[:, :1, :, :], sp1, sp2, sp3, sp4, sp5], 1)

        out1 = self.classifier_A(cat)
        out2 = self.classifier_V(cat)

        return cat, torch.sigmoid(out1), torch.sigmoid(out2)

class GUNet(torch.nn.Module):
    def __init__(self):
        super(GUNet, self).__init__()
        self.GUnet = gnn.GraphUNet(80+1, 32, 2, 3)

    def forward(self, x, edge_index):
        x = self.GUnet(x, edge_index)
        return x
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
    
        self.GUnet_A = GUNet()
        self.GUnet_V = GUNet()

    def get_edge_inform(self, data, search_range):
        cur_mask = data[0][0].clone()
        idx_mask = (cur_mask - 1).long()
        # idx_mask = torch.repeat_interleave(torch.unsqueeze(cur_mask, 2),2,2).long()
        vessel_idx = torch.nonzero(cur_mask, as_tuple=True)
        # cur_idx_mask_idx = torch.nonzero(cur_mask, as_tuple=False)
        idx_mask[vessel_idx] = torch.arange(0, len(vessel_idx[0]), 1).long().cuda(1)
        xx, yy = np.meshgrid(np.linspace(-(search_range // 2), search_range // 2, search_range),
                             np.linspace(-(search_range // 2), search_range // 2, search_range))
        xx = np.array(xx, dtype=np.int).reshape([-1])
        yy = np.array(yy, dtype=np.int).reshape([-1])
        edge_idx1 = torch.tensor([]).long().cuda(1)
        edge_idx2 = torch.tensor([]).long().cuda(1)
        for x, y in zip(xx, yy):
            cur_shift = torch.roll(cur_mask, (y, x), dims=(0, 1))
            cur_idx_mask_shift = torch.roll(idx_mask, (y, x), dims=(0, 1))
            conn_mask = (cur_mask - cur_shift) == 0
            cur_shift_idx = torch.nonzero(conn_mask[vessel_idx], as_tuple=True)
            cur_shift_pts = torch.nonzero(conn_mask[vessel_idx], as_tuple=False)
            cur_idx_mask_idx = cur_idx_mask_shift[vessel_idx][cur_shift_idx]
            edge_idx1 = torch.cat([edge_idx1, cur_shift_pts.view(-1)], 0)
            edge_idx2 = torch.cat([edge_idx2, cur_idx_mask_idx.view(-1)], 0)
        edge_index = torch.cat([torch.unsqueeze(edge_idx1, 0), torch.unsqueeze(edge_idx2, 0)], 0).long()
       

        return edge_index

    def forward(self, feature, cnn_data, mask):

        graph_feat = torch.masked_select(feature.cpu(), (mask[0][0]>0).cpu()).view([80+1, -1])

        edge_index_data = self.get_edge_inform(cnn_data, 5)
        x_data = graph_feat.transpose(1, 0).clone().cuda(1)

        gnn_data = Data(x=x_data, edge_index=edge_index_data)

        # GNN - Artery
        x_A = self.GUnet_A(gnn_data.x, gnn_data.edge_index)
        # GNN - Vein
        x_V = self.GUnet_V(gnn_data.x, gnn_data.edge_index)

        return torch.sigmoid(x_A), torch.sigmoid(x_V)

