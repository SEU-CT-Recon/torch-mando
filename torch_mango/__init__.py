__version__ = "0.0.1"

import torch

try:
    import torch_mango_cuda
except Exception as e:
    print("Importing exception")

from .utils import normalize_shape


class RadonForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *cfg):
        ctx.cfg = cfg
        sinogram = torch_mango_cuda.forward(x, *ctx.cfg)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        grad = torch_mango_cuda.backward(grad_x, *ctx.cfg)

        return grad, None


class RadonBackprojection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cfg):
        ctx.cfg = cfg
        image = torch_mango_cuda.backward(x, *ctx.cfg)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        grad = torch_mango_cuda.forward(grad_x, *ctx.cfg)

        return grad, None


class RadonFanbeam():
    def __init__(self, sid, sdd, views, detEleSize, oversample, startAngle, totalAngle, imgDim, imgPixSize, fpjStepSize,
                 sgmHeight, sgmWidth, reconKernelName, reconKernelParam, detOffcenter, imgRot, imgXCenter, imgYCenter):
        self.sid = sid
        self.sdd = sdd
        self.views = views
        self.detEleSize = detEleSize
        self.oversample = oversample
        self.startAngle = startAngle
        self.totalAngle = totalAngle
        self.imgDim = imgDim
        self.imgPixSize = imgPixSize
        self.fpjStepSize = fpjStepSize
        self.sgmHeight = sgmHeight
        self.sgmWidth = sgmWidth
        self.reconKernelName = reconKernelName
        self.reconKernelParam = reconKernelParam
        self.detOffcenter = detOffcenter
        self.imgRot = imgRot
        self.imgXCenter = imgXCenter
        self.imgYCenter = imgYCenter

    def _check_input(self, x, square=False):
        if not x.is_contiguous():
            x = x.contiguous()

        if square:
            assert x.size(1) == x.size(2), f"Input images must be square, got shape ({x.size(1)}, {x.size(2)})."

        return x

    @normalize_shape(2)
    def forward(self, x):
        r"""Radon forward projection.

        :param x: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        :returns: PyTorch GPU tensor containing sinograms. Has shape :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        """
        x = self._check_input(x, square=True)

        return RadonForward.apply(x, self.detOffcenter, self.sid, self.sdd, self.views, self.sgmWidth, self.oversample,
                                  self.startAngle, self.totalAngle, self.imgDim, self.imgPixSize, self.fpjStepSize)

    @normalize_shape(2)
    def backprojection(self, sinogram):
        r"""Radon backward projection.

        :param sinogram: PyTorch GPU tensor containing sinograms with shape  :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        :returns: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        """
        sinogram = self._check_input(sinogram)

        return RadonBackprojection.apply(sinogram, self.sgmHeight, self.sgmWidth, self.views, self.reconKernelName,
                                         self.reconKernelParam, self.totalAngle, self.detEleSize, self.detOffcenter,
                                         self.sid, self.sdd, self.imgDim, self.imgPixSize, self.imgRot, self.imgXCenter,
                                         self.imgYCenter)

    def backward(self, sinogram):
        r"""Same as backprojection."""
        return self.backprojection(sinogram)
