import torch.nn as nn
import torch
from torch_mando import *
import torch.nn.functional as F
from dft_recon import DFTParallelRecon_diff
import tifffile
import numpy as np
from crip.io import imwriteTiff, imreadRaw
from crip.preprocess import fanToPara
    

if __name__ == '__main__':
    cfg = MandoFanBeamConfig(imgDim=512, pixelSize=0.8, sid=800, sdd=1200, detEltCount=800, detEltSize=0.8,
                             views=720, reconKernelEnum=KERNEL_GAUSSIAN_RAMP, reconKernelParam=0.75, fpjStepSize=0.2)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    I = imreadRaw('./img_head.raw', 512, 512, dtype=np.float32)
    I = torch.FloatTensor(I.copy()[None, None, :]).cuda()
    with torch.no_grad():
        sgm = MandoFanbeamFpj(I, cfg).squeeze().detach().cpu().numpy()
    
    # re-binning fan beam projection to parallel projection
    det = (np.arange(cfg.detEltCount) - (cfg.detEltCount-1)/2) * cfg.detEltSize
    gammas = np.arctan2(det, cfg.sdd)
    betas = np.arange(cfg.views) / cfg.views * np.pi * 2 - np.pi / 2
    oThetas = [-np.pi, np.pi * 2 / cfg.views, np.pi]
    oLines = [np.min(det), det[1] - det[0], np.max(det) + det[1] - det[0]]
    
    sgm_parallel = fanToPara(sgm, gammas, -betas, cfg.sid, oThetas, oLines)
    sgm_parallel = np.flipud(sgm_parallel).copy()

    x = torch.FloatTensor(sgm_parallel).unsqueeze(0).unsqueeze(0).to(device)
    y = torch.FloatTensor(sgm).unsqueeze(0).unsqueeze(0).to(device)
    dftRecon = DFTParallelRecon_diff(cfg).to(device)
    with torch.no_grad():
        output_dft = dftRecon(x).detach().cpu().numpy()
        output_fbp = MandoFanbeamFbp(y, cfg).detach().cpu().numpy()
    tifffile.imwrite('./temp.tif', output_dft)
    tifffile.imwrite('./temp_fbp.tif', output_fbp)
    