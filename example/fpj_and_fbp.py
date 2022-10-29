import numpy as np
import torch
from crip.io import imwriteTiff
from torch_mango import MangoFanbeam

device = torch.device('cuda')
img = np.load("shepplogan.npy")  # 512*512
imgDim = img.shape[0]
views = 360
kernelId = 1  # Ramp Kernel
totalAngle = 360
detEleCount = 300
detEleSize = 1
imgPixelSize = 0.5
radon = MangoFanbeam(750, 800, views, detEleSize, 1, 0, totalAngle, imgDim, imgPixelSize, 0.2, views, detEleCount,
                     kernelId, 1, 0, 0, 0, 0, True)


def take(x):
    return x.detach().cpu().numpy()


with torch.no_grad():
    x = torch.FloatTensor(np.array([img, img.T]))  # B, H, W
    x = x.to(device)

    sinogram = radon.forward(x)
    imwriteTiff(take(x), 'img.tif')
    imwriteTiff(take(sinogram), 'sinogram.tif')
    fbp = radon.backward(sinogram)
    imwriteTiff(take(fbp), 'recon.tif')

print("FBP Error", torch.norm(x - fbp).item())
