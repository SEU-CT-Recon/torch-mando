import numpy as np
import torch
from crip.io import imwriteTiff
from crip.preprocess import binning
from torch_mango import RadonFanbeam

device = torch.device('cuda')
img = np.load("shepplogan.npy")  # 512*512
# img = binning(img, (16, 16))
print(img.shape)  # 32*32
imgDim = img.shape[0]
views = 360
Ramp = 1
totalAngle = 360
detEleCount = 300
detEleSize = 5
imgPixelSize = 2
radon = RadonFanbeam(750, 800, views, detEleSize, 1, 0, totalAngle, imgDim, imgPixelSize, 0.2, views, detEleCount,
                     Ramp, 1, 0, 0, 0, 0)


def take(x):
    return x.detach().cpu().numpy()


with torch.no_grad():
    x = torch.FloatTensor(np.array([img, img.T]))
    # B,H,W
    # x = x.unsqueeze(0)
    x = x.to(device)

    sinogram = radon.forward(x)
    imwriteTiff(take(x), 'img.tif')
    imwriteTiff(take(sinogram), 'sinogram.tif')
    # fbp = radon.backward(sinogram)
    # imwriteTiff(take(fbp), 'recon.tif')

# # print("FBP Error", torch.norm(x - fbp).item())
