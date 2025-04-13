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
import Myloss
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from math import log10, sqrt
import glob
import logging

logging.basicConfig(filename="gamma_DCE.log", filemode="a", format="%(levelname)s: %(message)s",level=logging.INFO)
logger=logging.getLogger()

torch.autograd.set_detect_anomaly(True)
 
def PSNR(original, compressed): 
	mse = np.mean((original - compressed) ** 2) 
	if mse == 0:  # MSE is zero means no noise is present in the signal
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	return psnr

def lowlight(image_path, label_path,DCE_net):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	data_lowlight = Image.open(image_path)
	label_lowlight = Image.open(label_path)

	# Normalize and prepare tensors for DCE network
	data_lowlight = (np.asarray(data_lowlight) / 255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()

	label_lowlight = (np.asarray(label_lowlight) / 255.0)
	label_lowlight = torch.from_numpy(label_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()
  
	enhanced_image, _ = DCE_net(data_lowlight)
	
	# Remove batch dimension and convert to numpy (H, W, C) format
	enhanced_imagenp = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
	label_lowlightnp = label_lowlight.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

	# Ensure the image is in range [0, 255] as uint8
	enhanced_imagenp = (enhanced_imagenp * 255).astype(np.uint8)
	label_lowlightnp = (label_lowlightnp * 255).astype(np.uint8)
	
	# Calculate PSNR and SSIM
	psnr = PSNR(enhanced_imagenp, label_lowlightnp)
	ssim = SSIM(enhanced_imagenp, label_lowlightnp, win_size=3, multichannel=True)
	# print("PSNR :", psnr, " SSIM :", ssim)
	return psnr, ssim

def test_lowlightimage(DCE_net,config,epoch):
	with torch.no_grad():
		sum_psnr = 0
		sum_ssim = 0
		test_size = 0
		filePath = config.test_image_path
		file_name = 'low'
		label_name = 'high'
		test_list = glob.glob(filePath + file_name + "/*")

		for image in test_list:
			image_label = image.replace(file_name, label_name)
			psnr, ssim = lowlight(image, image_label,DCE_net)
			test_size += 1
			sum_psnr += psnr
			sum_ssim += ssim
		logger.info("Epoch :"+ str(epoch) +", PSNR :" + str(sum_psnr / test_size) + ", SSIM :" + str(sum_ssim / test_size))
		print("avg PSNR :", sum_psnr / test_size, " avg SSIM:", sum_ssim / test_size)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	DCE_net = model.enhance_net_nopool().cuda()

	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()


	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image,A  = DCE_net(img_lowlight)

			Loss_TV = 200*L_TV(A)
			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

			loss_col = 5*torch.mean(L_color(enhanced_image))

			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			
			
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp
			#

			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				logger.info("Epoch :"+ str(epoch) +", Loss at iteration" + str(iteration+1) + ":" + str(loss.item()))
				print("Loss at iteration ", iteration+1, " : ", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
		test_lowlightimage(DCE_net,config,epoch) 		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default=r"E:/python/Zero-DCE-master/Zero-DCE_code/data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="E:/python/Zero-DCE-master/Zero-DCE_code/snapshots_gamma_DCE/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--test_image_path', type=str, default= "E:/python/Zero-DCE-master/Zero-DCE_code/data/Lol_v1_test/")
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	train(config)








	
