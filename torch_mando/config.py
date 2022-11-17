import torch


class MandoFanBeamConfig():
    def __init__(self, imgDim: int, pixelSize: float, sid: float, sdd: float, detEltCount: int, detEltSize: float, views: int,
                 reconKernelEnum: str, reconKernelParam: float, imgRot: float=0, imgXCenter: float=0, imgYCenter: float=0,
                 startAngle: float=0, totalScanAngle: float=360, detOffCenter: float=0, oversampleSize: int=2, 
                 fpjStepSize: float=0.2, fovCrop: bool=False):
        self.imgDim = imgDim                        # fpj & fbp
        self.pixelSize = pixelSize                  # fpj & fbp
        self.sid = sid                              # fpj & fbp
        self.sdd = sdd                              # fpj & fbp
        self.detEltCount = detEltCount              # fpj & fbp
        self.detEltSize = detEltSize                # fpj & fbp
        self.views = views                          # fpj & fbp
        self.sgmHeight = self.views                 # fbp
        self.sgmWidth = self.detEltCount            # fbp
        self.reconKernelEnum = reconKernelEnum      # fbp
        self.reconKernelParam = reconKernelParam    # fbp
        self.imgRot = imgRot                        # fbp
        self.imgXCenter = imgXCenter                # fbp
        self.imgYCenter = imgYCenter                # fbp
        self.startAngle = startAngle                # fpj
        self.totalScanAngle = totalScanAngle        # fpj & fbp
        self.detOffCenter = detOffCenter            # fpj & fbp
        self.oversampleSize = oversampleSize        # fpj
        self.fpjStepSize = fpjStepSize              # fpj
        self.fovCrop = fovCrop                      # fbp
        self._checkParams()
        # params file
        self.pmatrixFlag = False
        self.pmatrixFile = torch.empty(0)
        self.pmatrixDetEltSize = self.detEltSize  # for integrated CT with different detEltSize
        self.nonuniformSID = False
        self.sidFile = torch.empty(0)
        self.nonuniformSDD = False
        self.sddFile = torch.empty(0)
        self.nonZeroSwingAngle = False
        self.swingAngleFile = torch.empty(0)
        self.nonuniformScanAngle = False
        self.scanAngleFile = torch.empty(0)
        self.nonuniformOffCenter = False
        self.offCenterFile = torch.empty(0)
        
    def _checkParams(self):
        assert self.imgDim > 0, "Image dimension must be positive."
        assert self.pixelSize > 0, "Pixel size must be positive."
        assert self.sid > 0, "Source to isocenter distance must be positive."
        assert self.sdd > 0, "Source to detector distance must be positive."
        assert self.detEltCount > 0, "Detector element count must be positive."
        assert self.detEltSize > 0, "Detector element size must be positive."
        assert self.views > 0, "Number of views must be positive."

        assert isinstance(self.oversampleSize, int), "Oversample size must be int."
        assert self.fpjStepSize > 0, "FPJ step size must be positive."
        assert self.fovCrop in [True, False], "FOV crop must be bool."
    
    def addPmatrixFile(self, array: torch.Tensor, pmatrixDetEltSize: float=-1):
        self.pmatrixFlag = True
        self.pmatrixFile = array
        if pmatrixDetEltSize == -1:
            self.pmatrixDetEltSize = self.detEltSize  # for integrated CT with different detEltSize
        else:
            self.pmatrixDetEltSize = pmatrixDetEltSize
    
    def addSIDFile(self, array: torch.Tensor):
        self.nonuniformSID = True
        self.sidFile = array

    def addSDDFile(self, array: torch.Tensor):
        self.nonuniformSDD = True
        self.sddFile = array
    
    def addScanAngleFile(self, array: torch.Tensor):
        self.nonuniformScanAngle = True
        self.scanAngleFile = array
        
    def addDetectorOffCenterFile(self, array: torch.Tensor):
        self.nonuniformOffCenter = True
        self.offCenterFile = array
