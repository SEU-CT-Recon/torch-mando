import numpy as np
import torch
from crip.io import imwriteTiff
from torch_mando import MandoFanbeamFbp, MandoFanbeamFpj, MandoFanBeamConfig, KERNEL_RAMP


def take(x):
    return x.detach().cpu().numpy()


device = torch.device('cuda')
img = np.load("shepplogan.npy")  # 512*512
imgDim = img.shape[0]
views = 360
totalAngle = 360
detEleCount = 650
detEleSize = 1
imgPixelSize = 0.5
cfg = MandoFanBeamConfig(imgDim=imgDim, pixelSize=imgPixelSize, sid=750, sdd=1250, detEltCount=detEleCount,
                         detEltSize=detEleSize, views=views, reconKernelEnum=KERNEL_RAMP, reconKernelParam=1,
                         fovCrop=False)

img1, img2, img3, img4 = img, img.T, np.fliplr(img), np.flipud(img)
with torch.no_grad():
    x = torch.FloatTensor(np.array([[img1, img2], [img3, img4]]))  # B, C, H, W
    x = x.to(device)

    sinogram = MandoFanbeamFpj(x, cfg)
    imwriteTiff(take(x), 'img.tif')
    imwriteTiff(take(sinogram), 'sinogram.tif')
    recon = MandoFanbeamFbp(sinogram, cfg)
    imwriteTiff(take(recon), 'recon.tif')

print("FBP Error", torch.norm(x - recon).item())
