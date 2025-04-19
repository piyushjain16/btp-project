import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ DecomNet ------------------
class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        self.conv0 = nn.Conv2d(4, 64, kernel_size=9, padding=4)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 4, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_rgb, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat((x, max_rgb), dim=1)

        x = self.relu(self.conv0(x_cat))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.sigmoid(self.conv6(x))

        R = x[:, :3, :, :]
        I = x[:, 3:, :, :]
        return R, I

# ------------------ BCE-Net ------------------
class BceNet(nn.Module):
    def __init__(self, t=12):  # t is output channel (e.g. 1 for binary, 2+ for multi-class)
        super(BceNet, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, t, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, I):
        out0 = self.relu(self.conv0(I))     # (B,32,H,W)
        out1 = self.relu(self.conv1(out0))  # (B,32,H,W)
        out2 = self.relu(self.conv2(out1))  # (B,32,H,W)
        out3 = self.relu(self.conv3(out2))  # (B,32,H,W)

        cat4 = torch.cat([out2, out3], dim=1)     # (B,64,H,W)
        out4 = self.relu(self.conv4(cat4))        # (B,32,H,W)

        cat5 = torch.cat([out1, out4], dim=1)     # (B,64,H,W)
        out5 = self.relu(self.conv5(cat5))        # (B,32,H,W)

        cat6 = torch.cat([out0, out5], dim=1)     # (B,64,H,W)
        out6 = self.conv6(cat6)                   # (B,t,H,W)

        x_r = 0.5+ 2*F.tanh(out6)
        r1,r2,r3,r4 = torch.split(x_r, 3, dim=1)
        beta=0.5

        x = torch.clamp(I, min=1e-8)
        x = torch.pow(x,r1) + beta * (x - torch.pow(x,r1))
        x = torch.pow(x,r2) + beta * (x - torch.pow(x,r2))
        x = torch.pow(x,r3) + beta * (x - torch.pow(x,r3))
        x = torch.pow(x,r4) + beta * (x - torch.pow(x,r4))
# 		x = torch.pow(x,r5) + beta * (x - torch.pow(x,r5))
# 		x = torch.pow(x,r6) + beta * (x - torch.pow(x,r6))
# 		x = torch.pow(x,r7) + beta * (x - torch.pow(x,r7))
# 		x = torch.pow(x,r8) + beta * (x - torch.pow(x,r8))
        r = torch.cat([r1,r2,r3,r4],1)

        return x,r

class FullModel(nn.Module):
    def __init__(self, t=12):
        super(FullModel, self).__init__()
        self.decom_net = DecomNet()
        self.bce_net = BceNet(t)

    def forward(self, x):
        R, I = self.decom_net(x)
        out, gamma = self.bce_net(I)
        enhanced_image = R * out 
        return R, I, out, gamma,enhanced_image



# def __init__(self, num_layers = 5):
#         super(DecomNet, self).__init__()
#         layerlist = [
#                     nn.Conv2d(4, 64, 9, stride=1, padding=4, bias=False),
#                     nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
#                     nn.ReLU(True)
#             ]
#         for l in range(num_layers):
#             layerlist.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False))
#             layerlist.append(nn.BatchNorm2d(64, momentum=0.95, eps=1e-03))
#             layerlist.append(nn.ReLU(True))
#         layerlist.append(nn.Conv2d(64, 4, 3, stride=1, padding=1, bias=False))
#         layerlist.append(nn.Sigmoid())
#         self.model = nn.Sequential(*layerlist)

#     def forward(self, input):
#         input_max = input.max(1)[0].unsqueeze(1)
#         input = torch.cat((input,input_max), 1)
#         outp = self.model(input)
#         R = outp[:,0:3,:,:]
#         I = outp[:,3:4,:,:]
#         return R, I

# class enhance_net_nopool(nn.Module):

# 	def __init__(self):
# 		super(enhance_net_nopool, self).__init__()

# 		self.relu = nn.ReLU(inplace=True)

# 		number_f = 32
# 		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
# 		self.sigmoid = nn.Sigmoid(inplace=True)

#         self.r_conv1 = nn.Conv2d(4, number_f*2, 9, 1, 4, bias=False)
# 	    self.r_conv2 = nn.Conv2d(number_f*2, number_f*2, 3, 1, 1, bias=False)
#         self.r_conv3 = nn.Conv2d(number_f*2, number_f*2, 3, 1, 1, bias=False)
#         self.r_conv4 = nn.Conv2d(number_f*2, number_f*2, 3, 1, 1, bias=False)
#         self.r_conv5 = nn.Conv2d(number_f*2, number_f*2, 3, 1, 1, bias=False)
#         self.r_conv6 = nn.Conv2d(number_f*2, 4, 3, 1, 1, bias=False)


# 		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
# 		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
# 		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
# 		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
# 		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
# 		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
# 		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)

# 		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
# 		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

# 	def forward(self, x):

# 		x1 = self.relu(self.e_conv1(x))
# 		# p1 = self.maxpool(x1)
# 		x2 = self.relu(self.e_conv2(x1))
# 		# p2 = self.maxpool(x2)
# 		x3 = self.relu(self.e_conv3(x2))
# 		# p3 = self.maxpool(x3)
# 		x4 = self.relu(self.e_conv4(x3))

# 		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
# 		# x5 = self.upsample(x5)
# 		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

# 		x_r = 0.5+ 2*F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
# 		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
# 		beta=0.5

# 		x = torch.clamp(x, min=1e-8)
# 		x = torch.pow(x,r1) + beta * (x - torch.pow(x,r1))
# 		x = torch.pow(x,r2) + beta * (x - torch.pow(x,r2))
# 		x = torch.pow(x,r3) + beta * (x - torch.pow(x,r3))
# 		x = torch.pow(x,r4) + beta * (x - torch.pow(x,r4))
# 		x = torch.pow(x,r5) + beta * (x - torch.pow(x,r5))
# 		x = torch.pow(x,r6) + beta * (x - torch.pow(x,r6))
# 		x = torch.pow(x,r7) + beta * (x - torch.pow(x,r7))
# 		x = torch.pow(x,r8) + beta * (x - torch.pow(x,r8))
# 		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)

# 		return x,r