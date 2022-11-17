import numpy as np
import torch
from crip.io import imwriteTiff
from torch_mando import MandoFanbeamFbp, MandoFanbeamFpj, MandoFanBeamConfig, KERNEL_GAUSSIAN_RAMP
import json


def take(x):
    return x.detach().cpu().numpy()


def readParamsFile(path):
    with open(path, 'r') as f:
        file = json.loads(f.read())
    return torch.tensor(file['Value'])


device = torch.device('cuda')
sgm = np.fromfile('./sgm/sgm.raw', dtype=np.float32).reshape(495, 5120)
imgDim = 512
views = 495
totalAngle = -197.28
detEleCount = 5120
detEleSize = 0.1
imgPixelSize = 240 / 512
cfg = MandoFanBeamConfig(imgDim=imgDim, pixelSize=imgPixelSize, sid=749.78, sdd=1060, detEltCount=detEleCount, 
                         detEltSize=detEleSize, views=views, reconKernelEnum=KERNEL_GAUSSIAN_RAMP, reconKernelParam=4, 
                         imgRot=0, detOffCenter=16.47, startAngle=0, totalScanAngle=totalAngle, fovCrop=False)
# add params file
cfg.addPmatrixFile(readParamsFile('./params/pmatrix_file.jsonc'), pmatrixDetEltSize=0.4)
cfg.addSIDFile(readParamsFile('./params/sid_file.jsonc'))
cfg.addSDDFile(readParamsFile('./params/sdd_file.jsonc'))
cfg.addScanAngleFile(readParamsFile('./params/scan_angle.jsonc'))
cfg.addDetectorOffCenterFile(readParamsFile('./params/offcenter_file.jsonc'))

with torch.no_grad():
    sgm = torch.FloatTensor(np.array([sgm]))  # B, H, W
    sgm = sgm.to(device)

    img = MandoFanbeamFbp(sgm, cfg)
    imwriteTiff(take(img), 'recon.tif')
    re_sgm = MandoFanbeamFpj(img, cfg)
    imwriteTiff(take(re_sgm), 'sinogram.tif')

print("FBP Error", torch.norm(sgm - re_sgm).item())
