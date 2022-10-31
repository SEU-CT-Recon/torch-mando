from .utils import normalize_shape
from .config import MandoFanBeamConfig
import torch

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
    'MandoFanBeamConfig', 'KERNEL_NONE', 'KERNEL_RAMP', 'KERNEL_HAMMING', 'KERNEL_GAUSSIAN_RAMP'
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