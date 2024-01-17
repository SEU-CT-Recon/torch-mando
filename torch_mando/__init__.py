from .utils import normalize_shape
from .config import MandoFanBeamConfig
import torch
import torch.nn.functional as F

try:
    import torch_mando_cuda
    print('[torch-mando / torch_mando_cuda] Using torch_mando_cuda backend.')
except Exception as e:
    print("[torch-mando / torch_mando_cuda] Failed to import torch_mando_cuda!")
    print(e)
    exit(1)



__version__ = "0.0.1"
__all__ = [
    'MandoFanbeamFpj', 'MandoFanbeamFbp', 'MandoFanbeamFpjLayer', 'MandoFanbeamFbpLayer', 
    'MandoFanBeamConfig', 'KERNEL_NONE', 'KERNEL_RAMP', 'KERNEL_HAMMING', 'KERNEL_GAUSSIAN_RAMP',
    'MandoFanbeamFbpLayerNext'
]

KERNEL_NONE = 0
KERNEL_RAMP = 1
KERNEL_HAMMING = 2
KERNEL_GAUSSIAN_RAMP = 4
        
        
class FanBeamFPJ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cfg: MandoFanBeamConfig):
        ctx.cfg = cfg
        sinogram = torch_mando_cuda.forward(x, cfg.detOffCenter, cfg.sid, cfg.sdd, cfg.views, cfg.detEltCount, 
                                            cfg.detEltSize, cfg.oversampleSize, cfg.startAngle, cfg.totalScanAngle, 
                                            cfg.imgDim, cfg.pixelSize, cfg.fpjStepSize,
                                            cfg.pmatrixFlag, cfg.pmatrixFile.clone(), cfg.pmatrixDetEltSize, 
                                            cfg.nonuniformSID, cfg.sidFile.clone(), cfg.nonuniformSDD, cfg.sddFile.clone(), 
                                            cfg.nonuniformScanAngle, cfg.scanAngleFile.clone(),  # clone() is necessary
                                            cfg.nonuniformOffCenter, cfg.offCenterFile.clone())

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        cfg: MandoFanBeamConfig = ctx.cfg
        grad = torch_mando_cuda.backward(grad_x, cfg.sgmHeight, cfg.sgmWidth, cfg.views, cfg.reconKernelEnum, 
                                         cfg.reconKernelParam, cfg.totalScanAngle, cfg.detEltSize, cfg.detOffCenter, 
                                         cfg.sid, cfg.sdd, cfg.imgDim, cfg.pixelSize, cfg.imgRot, cfg.imgXCenter, cfg.imgYCenter, 
                                         cfg.pmatrixFlag, cfg.pmatrixFile.clone(), cfg.pmatrixDetEltSize, 
                                         cfg.nonuniformSID, cfg.sidFile.clone(), cfg.nonuniformSDD, cfg.sddFile.clone(), 
                                         cfg.nonuniformScanAngle, cfg.scanAngleFile.clone(),  # clone() is necessary
                                         cfg.nonuniformOffCenter, cfg.offCenterFile.clone(), cfg.fovCrop)

        return grad, None


class FanBeamFBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cfg: MandoFanBeamConfig):
        ctx.cfg = cfg
        image = torch_mando_cuda.backward(x, cfg.sgmHeight, cfg.sgmWidth, cfg.views, cfg.reconKernelEnum, 
                                          cfg.reconKernelParam, cfg.totalScanAngle, cfg.detEltSize, cfg.detOffCenter, 
                                          cfg.sid, cfg.sdd, cfg.imgDim, cfg.pixelSize, cfg.imgRot, cfg.imgXCenter, cfg.imgYCenter, 
                                          cfg.pmatrixFlag, cfg.pmatrixFile.clone(), cfg.pmatrixDetEltSize, 
                                          cfg.nonuniformSID, cfg.sidFile.clone(), cfg.nonuniformSDD, cfg.sddFile.clone(), 
                                          cfg.nonuniformScanAngle, cfg.scanAngleFile.clone(),  # clone() is necessary
                                          cfg.nonuniformOffCenter, cfg.offCenterFile.clone(), cfg.fovCrop)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        cfg: MandoFanBeamConfig = ctx.cfg
        grad = torch_mando_cuda.forward(grad_x, cfg.detOffCenter, cfg.sid, cfg.sdd, cfg.views, cfg.detEltCount, 
                                        cfg.detEltSize, cfg.oversampleSize, cfg.startAngle, cfg.totalScanAngle, 
                                        cfg.imgDim, cfg.pixelSize, cfg.fpjStepSize,
                                        cfg.pmatrixFlag, cfg.pmatrixFile.clone(), cfg.pmatrixDetEltSize, 
                                        cfg.nonuniformSID, cfg.sidFile.clone(), cfg.nonuniformSDD, cfg.sddFile.clone(), 
                                        cfg.nonuniformScanAngle, cfg.scanAngleFile.clone(),  # clone() is necessary
                                        cfg.nonuniformOffCenter, cfg.offCenterFile.clone())

        return grad, None


@normalize_shape(2)
def MandoFanbeamFpj(img, cfg: MandoFanBeamConfig):
    r"""Radon forward projection.

    :param img: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
        given to the constructor of this class.
    :returns: PyTorch GPU tensor containing sinograms. Has shape :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
    """
    assert img.size(1) == img.size(2), f"Input images must be square, got ({img.size(1)}, {img.size(2)})."

    if not img.is_contiguous():
        img = img.contiguous()

    return FanBeamFPJ.apply(img, cfg)


@normalize_shape(2)
def MandoFanbeamFbp(sinogram, cfg: MandoFanBeamConfig):
    r"""Radon backward projection.

    :param sinogram: PyTorch GPU tensor containing sinograms with shape  :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
    :returns: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
        given to the constructor of this class.
    """
    if not sinogram.is_contiguous():
        sinogram = sinogram.contiguous()

    return FanBeamFBP.apply(sinogram, cfg)


class MandoFanbeamFpjLayer(torch.nn.Module):
    def __init__(self, cfg: MandoFanBeamConfig) -> None:
        super().__init__()

        self.fn = FanBeamFPJ.apply
        self.cfg = cfg

    def forward(self, x):
        return self.fn(x, self.cfg)


class MandoFanbeamFbpLayer(torch.nn.Module):
    def __init__(self, cfg: MandoFanBeamConfig) -> None:
        super().__init__()

        self.fn = FanBeamFBP.apply
        self.cfg = cfg

    def forward(self, x):
        return self.fn(x, self.cfg)
    

def zxInitRampFilter(N, du):
    ''' N: detector element count; du: detector element size
        output: (N)
    '''
    rampFilter = torch.zeros((N), dtype=torch.float32)
    center = N // 2 - 1
    for i in range(N):
        if (i - center) % 2 != 0:
            rampFilter[i] = -1 / ((torch.pi * (i - center) * du)**2)

    rampFilter[center] = 1 / ((2 * du)**2)

    return rampFilter


def zxInitGaussianFilter(N, sigma, du):
    ''' N: detector element count; sigma [pixel]; du: detector element size
    '''
    gaussianFilter = torch.zeros((N), dtype=torch.float32)
    center = N // 2 - 1
    for i in range(N):
        gaussianFilter[i] = torch.exp(-(i - center) ** 2 / (2 * sigma ** 2))

    return gaussianFilter / torch.sum(gaussianFilter) / du # /du to cancel out the *du in zxFilterSinogram
        

def zxFilterSinogram(sgm, reconKernel, du):
    ''' Filter each row of sinogram.
        sgm (B, H, W)
        reconKernel (N)
        du: detector element size
    '''
    reconKernel = reconKernel.view(1, 1, -1).to(sgm.device) # F.conv1d requires 3D kernel [C, H, W]

    oldShape = sgm.shape
    if len(sgm.shape) == 3:
        B, H, W = sgm.shape
        sgm = sgm.view(B, 1, H, W) # F.conv1d requires BCHW
    else:
        B, _, H, W = sgm.shape

    res = torch.zeros_like(sgm).to(sgm.device)
    for b in range(B):
        for h in range(H):
            res[b, :, h, :] = F.conv1d(sgm[b, :, h, :], reconKernel, padding='same')
    res *= du

    return res.view(oldShape)


class MandoFanbeamFbpLayerNext(torch.nn.Module):

    def __init__(self, cfg: MandoFanBeamConfig) -> None:
        super().__init__()

        self.fn = FanBeamFBP.apply
        self.filters = []

        if cfg.reconKernelEnum == KERNEL_GAUSSIAN_RAMP:
            self.filters.append(zxInitGaussianFilter(cfg.detEltCount, cfg.reconKernelParam))
            self.filters.append(zxInitRampFilter(cfg.detEltCount, cfg.detEltSize))
            
        if cfg.reconKernelEnum == KERNEL_RAMP:
            self.filters.append(zxInitRampFilter(cfg.detEltCount, cfg.detEltSize))

        if cfg.reconKernelEnum == KERNEL_HAMMING:
            raise NotImplementedError('Hamming filter is not implemented yet. If you use Ramp filter, please set reconKernelEnum to `KERNEL_RAMP`.')
        
        # self.cfg = { **cfg, 'reconKernelEnum': KERNEL_NONE, 'reconKernelParam': 0 }
        self.cfg = cfg.copy()
        self.cfg.reconKernelEnum = KERNEL_NONE
        self.cfg.reconKernelParam = 0

    def forward(self, x):
        for f in self.filters:
            x = zxFilterSinogram(x, f, self.cfg.detEltSize)

        return self.fn(x, self.cfg)
