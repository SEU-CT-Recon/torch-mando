import torch
from torch_mando import *
import torch.nn as nn
import torch.nn.functional as F


class DFTParallelRecon_diff(nn.Module):
    def __init__(self, cfg: MandoFanBeamConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.N = self.cfg.detEltCount
        self.a = self.cfg.detEltSize
        self.FN = self.N * 4
        self.FM = 2048
        self.M = self.cfg.imgDim
        self.V = self.cfg.views
        
    def forward(self, sgm):
        B, C, H, W = sgm.shape
        rec = torch.zeros((B * C, self.M, self.M), dtype=torch.float32).cuda()
        for i, s in enumerate(sgm.reshape(-1, H, W)):
            rec[i] = self.dft_recon(s, self.N, self.a, self.FN, self.FM, self.M, self.V)
    
        return rec.reshape(B, C, self.M, self.M)
        
    def dft_recon(self, sgm: torch.tensor, N: int, a: float, FN: int, FM: int, M: int, V: int):
        ### 1. sgm-1D dft
        R_k = torch.zeros((self.V, self.FN), dtype=torch.complex64).cuda()
        for j in range(V):
            R_k[j] = self.fft_trans(sgm[j, ...], a, N, FN)
        
        ### 1.1 sgm-1D dft filter, gaussianApodizedRamp freq domain
        if self.cfg.reconKernelEnum == KERNEL_GAUSSIAN_RAMP:
            R_k = self.fft_sgm_filter(R_k, a, self.cfg.reconKernelParam, FN)
        
        ### 2. interpolate
        # src (useless)
        # vec_k = (np.arange(FN) - (FN-1) / 2) * (1 / (FN*a))  # sgm
        # theta_ = np.arange(V) / V * 2 * np.pi - 2 * np.pi
        # m_theta, m_k = np.meshgrid(theta_, vec_k)
        # dst
        vec_k_f = (torch.arange(FM) - (FM-1) / 2) * (1 / (FM*self.cfg.pixelSize))  # freq
        vec_ky, vec_kx = torch.meshgrid(vec_k_f, vec_k_f, indexing='xy')
        m_k_f = torch.abs(vec_kx + 1j * vec_ky)
        m_theta_f = torch.arctan2(vec_ky, vec_kx)
        # move coordinate
        theta = (m_theta_f + 2 * torch.pi) / (2 * torch.pi) * V
        k = m_k_f * (FN*a) + (FN-1) / 2
        # adj for F.grid_sample
        theta_norm = (theta - V) / V
        k_norm = (k - (FN-1) / 2) / (FN // 2)
        
        R_k = torch.cat((R_k, R_k), axis=0)
        R_k_real = R_k.real[None, None, ...]
        R_k_imag = R_k.imag[None, None, ...]
        grid = torch.stack((k_norm, theta_norm), axis=-1)[None, ...].cuda()

        f_k_real = F.grid_sample(R_k_real, grid, mode='bilinear', align_corners=True).squeeze()
        f_k_imag = F.grid_sample(R_k_imag, grid, mode='bilinear', align_corners=True).squeeze()

        ### 3. img-2D dft
        f_n = self.fft2_trans(f_k_real + 1j * f_k_imag, self.cfg.pixelSize, FM)
        f_n = f_n[(FM-M)//2: (FM+M)//2, (FM-M)//2: (FM+M)//2]
        
        return f_n.real.to(torch.float32)
    
    def fft_trans(self, R: torch.tensor, a: float, N: int, FN: int):
        fm = R
        delta_x = a
        delta_k = 1 / (FN * delta_x)
        x_0 = - (N-1) / 2 * delta_x
        k_0 = - (FN-1) / 2 * delta_k
        m = torch.arange(N).cuda()
        n = torch.arange(FN).cuda()
        
        gm = fm * torch.exp(-1j * 2 * torch.pi * k_0 * delta_x * m)
        gn = torch.fft.fft(gm, n=FN, norm='backward')
        return delta_x * torch.exp(-1j * 2 * torch.pi * x_0 * (delta_k * n + k_0)) * gn

    def fft_sgm_filter(self, R: torch.tensor, a: float, sigma: float, FN: int):
        fm = R
        n = (torch.arange(FN).cuda() - (FN-1) / 2) / (FN * a)
        kernel = torch.exp(-2 * torch.pi**2 * sigma**2 * a**2 * n**2)
        return kernel * fm
        
    def fft2_trans(self, fm2: torch.tensor, delta_x: float, FM: int):
        delta_k = 1 / (FM * delta_x)
        delta_kx = delta_ky = delta_k
        x_0 = y_0 = - (FM-1) / 2 * delta_x
        k_x0 = k_y0 = - (FM-1) / 2 * delta_k
        m_kx, m_ky = torch.meshgrid(torch.arange(FM), torch.arange(FM), indexing='xy')
        n_x, n_y = torch.meshgrid(torch.arange(FM), torch.arange(FM), indexing='xy')
        
        gm2 = fm2 * torch.exp(1j * 2 * torch.pi * x_0 * m_kx.cuda() * delta_kx) * \
                    torch.exp(1j * 2 * torch.pi * y_0 * m_ky.cuda() * delta_ky)
        gn2 = torch.fft.ifft2(gm2, s=(FM, FM), norm='forward')
        return delta_kx * delta_kx * torch.exp(1j * 2 * torch.pi * (delta_x * n_x.cuda() + x_0) * k_x0) * \
                                    torch.exp(1j * 2 * torch.pi * (delta_x * n_y.cuda() + y_0) * k_y0) * gn2