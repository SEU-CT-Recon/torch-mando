import numpy as np
import torch
from crip.io import imwriteTiff
from torch_mango import RadonFanbeam

device = torch.device('cuda:3')
img = np.load("shepplogan.npy")
image_size = img.shape[0]
views = 360
RAMP = 1
radon = RadonFanbeam(750, 1250, 360, 2, 1, 0, 360, 512, 1, 0.2, 360, 648, RAMP, 1, 0, 0, 0, 0)


def take(x):
    return x.detach().cpu().numpy()


with torch.no_grad():
    x = torch.FloatTensor(img)
    # B,H,W
    x = x.unsqueeze(0)
    x = x.to(device)
    
    sinogram = radon.forward(x)
    imwriteTiff(take(x), 'img.tif')
    imwriteTiff(take(sinogram), 'sinogram.tif')
    fbp = radon.backward(sinogram)
    imwriteTiff(take(fbp), 'recon.tif')

print("FBP Error", torch.norm(x - fbp).item())
