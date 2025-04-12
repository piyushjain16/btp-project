import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2
from skimage.metrics import structural_similarity as SSIM
from math import log10, sqrt

def PSNR(original, compressed): 
    original = np.array(original)
    compressed = np.array(compressed)
    mse = np.mean((original - compressed) ** 2) 
    if mse == 0:  # MSE is zero means no noise is present in the signal
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

def lowlight(image_path, label_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    label_lowlight = Image.open(label_path)

    # Normalize and prepare tensors for DCE network
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()

    label_lowlight = (np.asarray(label_lowlight) / 255.0)
    label_lowlight = torch.from_numpy(label_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('E:/python/Zero-DCE-master/Zero-DCE_code/snapshots/Epoch99.pth', weights_only=True))
  
    _,enhanced_image, _ = DCE_net(data_lowlight)
    
    # Remove batch dimension and convert to numpy (H, W, C) format
    enhanced_imagenp = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    label_lowlightnp = label_lowlight.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # Ensure the image is in range [0, 255] as uint8
    enhanced_imagenp = (enhanced_imagenp * 255).astype(np.uint8)
    label_lowlightnp = (label_lowlightnp * 255).astype(np.uint8)
    
    # Calculate PSNR and SSIM
    psnr = PSNR(enhanced_imagenp, label_lowlightnp)
    ssim = SSIM(enhanced_imagenp, label_lowlightnp, win_size=3, multichannel=True)
    print("PSNR :", psnr, " SSIM :", ssim)
    result_path = image_path.replace('low', 'result')

    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(result_path)):
      os.makedirs(os.path.dirname(result_path))
    # Save enhanced image using torchvision
    enhanced_imagenp = enhanced_imagenp.astype(np.float32) / 255.0  # Normalize to [0, 1]
    torchvision.utils.save_image(torch.from_numpy(enhanced_imagenp).permute(2, 0, 1), result_path)

    return psnr, ssim

if __name__ == '__main__':
    with torch.no_grad():
        sum_psnr = 0
        sum_ssim = 0
        test_size = 0
        filePath = 'E:/python/Zero-DCE-master/Zero-DCE_code/data/eval15/'
        file_name = 'low'
        label_name = 'high'
        test_list = glob.glob(filePath + file_name + "/*")

        for image in test_list:
            image_label = image.replace(file_name, label_name)
            psnr, ssim = lowlight(image, image_label)
            test_size += 1
            sum_psnr += psnr
            sum_ssim += ssim
        
        print("avg PSNR :", sum_psnr / test_size, " avg SSIM:", sum_ssim / test_size)

 
# def lowlight(image_path):
# 	os.environ['CUDA_VISIBLE_DEVICES']='0'
# 	data_lowlight = Image.open(image_path)

 

# 	data_lowlight = (np.asarray(data_lowlight)/255.0)


# 	data_lowlight = torch.from_numpy(data_lowlight).float()
# 	data_lowlight = data_lowlight.permute(2,0,1)
# 	data_lowlight = data_lowlight.cuda().unsqueeze(0)

# 	DCE_net = model.enhance_net_nopool().cuda()
# 	DCE_net.load_state_dict(torch.load('Zero-DCE-master\Zero-DCE_code\snapshots\Epoch99.pth'))
# 	start = time.time()
# 	_,enhanced_image,_ = DCE_net(data_lowlight)

# 	end_time = (time.time() - start)
# 	print(end_time)
# 	image_path = image_path.replace('test_data','result')
# 	result_path = image_path
# 	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
# 		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

# 	torchvision.utils.save_image(enhanced_image, result_path)

# if __name__ == '__main__':
# # test_images
# 	with torch.no_grad():
# 		filePath = 'E:/python/Zero-DCE-master/Zero-DCE_code/data/test_data/'
	
# 		file_list = os.listdir(filePath)

# 		for file_name in file_list:
# 			test_list = glob.glob(filePath+file_name+"/*") 
# 			for image in test_list:
# 				# image = image
# 				print(image)
# 				lowlight(image)

