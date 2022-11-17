'''
    This script trains a denoise network that uses noisy sinograms as input and clean reconstructed images as label
    using torch-mando.
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_mando import MandoFanbeamFbp, MandoFanbeamFpj, MandoFanBeamConfig, KERNEL_RAMP, MandoFanbeamFbpLayer
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from crip.io import listDirectory, imreadDicom, imwriteTiff, imreadTiff
from crip.lowdose import injectGaussianNoise
from tqdm import tqdm

views = 360
totalAngle = 360
detEleCount = 648
detEleSize = 0.8
imgPixelSize = 0.5
imgDim = 512
sid = 750
sdd = 1250
cfg = MandoFanBeamConfig(imgDim=imgDim, pixelSize=imgPixelSize, sid=sid, sdd=sdd, detEltCount=detEleCount,
                         detEltSize=detEleSize, views=views, reconKernelEnum=KERNEL_RAMP, reconKernelParam=1,
                         imgRot=0, detOffCenter=0, startAngle=0, totalScanAngle=totalAngle, fovCrop=False)

dataDir = '...'
noisyDir = '...'


def prepareNoisySinograms():
    print('Projecting noisy sinograms...')
    for path, file in listDirectory(dataDir, style='both'):
        img = imreadDicom(path) / 10
        img = injectGaussianNoise(img, 5)
        img = torch.from_numpy(img).to(torch.float32).cuda()
        sgm = MandoFanbeamFpj(img, cfg)
        imwriteTiff(sgm.detach().cpu().numpy(), os.path.join(noisyDir, file.replace('.IMA', '.tif')))

    print('Projecting done.')


class DeNoiseDataset(Dataset):
    def __init__(self, dir_):
        super().__init__()

        files = listDirectory(dir_, style='fullpath')
        noisyFiles = listDirectory(noisyDir, style='fullpath')
        self.cleans = torch.from_numpy(np.array([imreadDicom(x, np.float32) / 10 for x in files])).cuda()
        self.noisys = torch.from_numpy(np.array([imreadTiff(x, np.float32) for x in noisyFiles])).cuda()

    def __getitem__(self, idx):
        return self.noisys[idx].to(torch.float32), self.cleans[idx].to(torch.float32)

    def __len__(self):
        return len(self.cleans)


class MyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        # self.fbpLayer = MandoFanbeamFbpLayer(cfg) # If you prefer layer style.
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = MandoFanbeamFbp(x, cfg) # If you prefer functional style.
        # x = self.fbpLayer(x) # If you prefer layer style.
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


model = MyNet().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
epochLoss = AverageMeter()
nEpochs = 10
batchSize = 16
trainSet = DeNoiseDataset(dataDir)
trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, drop_last=True)

for epoch in range(nEpochs):
    with tqdm(total=(len(trainSet) - len(trainSet) % batchSize), ncols=80, desc='[Train]', ascii=True) as t:
        t.set_description('Epoch: {}/{}'.format(epoch + 1, nEpochs))

        for data in trainLoader:
            noisy, clean = data
            # Convert BHW to BCHW.
            noisy = noisy.unsqueeze(1).cuda()
            clean = clean.unsqueeze(1).cuda()

            pred = model(noisy).cuda()
            loss = criterion(pred, clean)
            epochLoss.update(torch.mean(loss))

            # Optimize the model.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epochLoss.avg))
            t.update(len(noisy))

    torch.save(model.state_dict(), f'./ckp/{epoch}.pth')
    print("Loss: ", epochLoss.avg)
