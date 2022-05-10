import pdb
import numpy as np

filename = 'unerf'
file = np.load(f"./render_output/surreal_{filename}_val/scores.npy", allow_pickle=True)

psnrs = file.item()['psnr']
ssims = file.item()['ssim']
fg_psnrs = file.item()['fg_psnr']
fg_ssims = file.item()['fg_ssim']

psnr =  [value for value in psnrs if not np.isnan(value)]
ssim =  [value for value in ssims if not np.isnan(value)]
fg_psnr =  [value for value in fg_psnrs if not np.isnan(value)]
fg_ssim =  [value for value in fg_ssims if not np.isnan(value)]

mean_psnr = np.mean(psnr)
mean_ssim = np.mean(ssim)
mean_fgpsnr = np.mean(fg_psnr)
mean_fgssim = np.mean(fg_ssim)

print("Mean PSNR: ", mean_psnr)
print("Mean SSIM: ", mean_ssim)
print("Mean FG PSNR: ", mean_fgpsnr)
print("Mean FG SSIM: ", mean_fgssim)
print("Nan PSNR: ", np.argwhere(np.isnan(psnrs)))
print("Nan SSIM: ", np.argwhere(np.isnan(ssims)))