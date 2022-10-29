import numpy as np
import torch
from crip.io import imwriteTiff
from torch_mango import MangoFanbeamFbp, MangoFanbeamFpj, MangoConfig, KERNEL_RAMP

device = torch.device('cuda')
img = np.load("shepplogan.npy")  # 512*512
imgDim = img.shape[0]
views = 360
totalAngle = 360
detEleCount = 650
detEleSize = 1
imgPixelSize = 0.5
cfg = MangoConfig(750, 1250, 0, totalAngle, views, 2, 0.2, views, detEleCount, imgDim, imgPixelSize, 0, 0, 0, True,
                  KERNEL_RAMP, 0, detEleSize, 0)

def take(x):
    return x.detach().cpu().numpy()


with torch.no_grad():
    x = torch.FloatTensor(np.array([img, img.T]))  # B, H, W
    x = x.to(device)

    sinogram = MangoFanbeamFpj(x, cfg)
    imwriteTiff(take(x), 'img.tif')
    imwriteTiff(take(sinogram), 'sinogram.tif')
    recon = MangoFanbeamFbp(sinogram, cfg)
    imwriteTiff(take(recon), 'recon.tif')

print("FBP Error", torch.norm(x - recon).item())
