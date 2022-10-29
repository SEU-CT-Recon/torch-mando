import torch

try:
    import torch_mango_cuda
    print('[torch-mango / torch_mango_cuda] Using torch_mango_cuda backend.')
except Exception as e:
    print("[torch-mango / torch_mango_cuda] Failed to import torch_mango_cuda!")
    print(e)
    exit(1)

from .utils import normalize_shape

__version__ = "0.0.1"
__all__ = [
    'MangoFanbeamFpj', 'MangoFanbeamFbp', 'MangoFanbeamFpjLayer', 'MangoFanbeamFbpLayer', 'MangoConfig', 'KERNEL_NONE',
    'KERNEL_RAMP', 'KERNEL_HAMMING', 'KERNEL_GAUSSIAN_RAMP'
]

KERNEL_NONE = 0
KERNEL_RAMP = 1
KERNEL_HAMMING = 2
KERNEL_GAUSSIAN_RAMP = 4


class MangoConfig():
    # Geometry
    sid: float
    sdd: float
    startAngle: float
    totalAngle: float

    # Projection
    views: int
    oversample: int
    fpjStepSize: float
    sgmHeight: int
    sgmWidth: int

    # Recon
    imgDim: int
    imgPixSize: float
    imgRot: float
    imgXCenter: float
    imgYCenter: float
    fovCrop: bool
    reconKernelEnum: int
    reconKernelParam: float

    # Detector
    detEleSize: float
    detOffcenter: float

    def __init__(self, sid, sdd, startAngle, totalAngle, views, oversample, fpjStepSize, sgmHeight, sgmWidth, imgDim,
                 imgPixSize, imgRot, imgXCenter, imgYCenter, fovCrop, reconKernelEnum, reconKernelParam, detEleSize,
                 detOffcenter):
        self.sid = sid
        self.sdd = sdd
        self.startAngle = startAngle
        self.totalAngle = totalAngle
        self.views = views
        self.oversample = oversample
        self.fpjStepSize = fpjStepSize
        self.sgmHeight = sgmHeight
        self.sgmWidth = sgmWidth
        self.imgDim = imgDim
        self.imgPixSize = imgPixSize
        self.imgRot = imgRot
        self.imgXCenter = imgXCenter
        self.imgYCenter = imgYCenter
        self.fovCrop = fovCrop
        self.reconKernelEnum = reconKernelEnum
        self.reconKernelParam = reconKernelParam
        self.detEleSize = detEleSize
        self.detOffcenter = detOffcenter


class RadonForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cfg: MangoConfig):
        ctx.cfg = cfg
        sinogram = torch_mango_cuda.forward(x, cfg.detOffcenter, cfg.sid, cfg.sdd, cfg.views, cfg.sgmWidth,
                                            cfg.detEleSize, cfg.oversample, cfg.startAngle, cfg.totalAngle, cfg.imgDim,
                                            cfg.imgPixSize, cfg.fpjStepSize)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        cfg: MangoConfig = ctx.cfg
        grad = torch_mango_cuda.backward(grad_x, cfg.sgmHeight, cfg.sgmWidth, cfg.views, cfg.reconKernelEnum,
                                         cfg.reconKernelParam, cfg.totalAngle, cfg.detEleSize, cfg.detOffcenter,
                                         cfg.sid, cfg.sdd, cfg.imgDim, cfg.imgPixSize, cfg.imgRot, cfg.imgXCenter,
                                         cfg.imgYCenter, cfg.fovCrop)

        return grad, None


class RadonBackprojection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cfg: MangoConfig):
        ctx.cfg = cfg
        image = torch_mango_cuda.backward(x, cfg.sgmHeight, cfg.sgmWidth, cfg.views, cfg.reconKernelEnum,
                                          cfg.reconKernelParam, cfg.totalAngle, cfg.detEleSize, cfg.detOffcenter,
                                          cfg.sid, cfg.sdd, cfg.imgDim, cfg.imgPixSize, cfg.imgRot, cfg.imgXCenter,
                                          cfg.imgYCenter, cfg.fovCrop)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        cfg: MangoConfig = ctx.cfg
        grad = torch_mango_cuda.forward(grad_x, cfg.detOffcenter, cfg.sid, cfg.sdd, cfg.views, cfg.sgmWidth,
                                        cfg.detEleSize, cfg.oversample, cfg.startAngle, cfg.totalAngle, cfg.imgDim,
                                        cfg.imgPixSize, cfg.fpjStepSize)

        return grad, None


@normalize_shape(2)
def MangoFanbeamFpj(img, cfg: MangoConfig):
    r"""Radon forward projection.

    :param img: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
        given to the constructor of this class.
    :returns: PyTorch GPU tensor containing sinograms. Has shape :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
    """
    assert img.size(1) == img.size(2), f"Input images must be square, got ({img.size(1)}, {img.size(2)})."

    if not img.is_contiguous():
        img = img.contiguous()

    return RadonForward.apply(img, cfg)


@normalize_shape(2)
def MangoFanbeamFbp(sinogram, cfg: MangoConfig):
    r"""Radon backward projection.

    :param sinogram: PyTorch GPU tensor containing sinograms with shape  :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
    :returns: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
        given to the constructor of this class.
    """
    if not sinogram.is_contiguous():
        sinogram = sinogram.contiguous()

    return RadonBackprojection.apply(sinogram, cfg)


class MangoFanbeamFpjLayer(torch.nn.Module):
    def __init__(self, cfg: MangoConfig) -> None:
        super().__init__()

        self.fn = RadonForward.apply
        self.cfg = cfg

    def forward(self, x):
        return self.fn(x, self.cfg)


class MangoFanbeamFbpLayer(torch.nn.Module):
    def __init__(self, cfg: MangoConfig) -> None:
        super().__init__()

        self.fn = RadonBackprojection.apply
        self.cfg = cfg

    def forward(self, x):
        return self.fn(x, self.cfg)